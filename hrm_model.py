"""HRM-style char-level reasoning model.

A tiny, from-scratch re-implementation of the architectural ideas behind
`sapientinc/HRM-Text-1B`:

  * A dual-timescale recurrent reasoning core: two transformer stacks, a slow
    high-level module (H) and a fast low-level module (L), iterated
    ``H_cycles x L_cycles`` times over the same embeddings with additive state
    injection (``z_L + z_H``). This buys extra "compute depth" at fixed params.
  * Modern blocks: RMSNorm (parameterless, pre-norm), RoPE, SwiGLU, gated MHA,
    scaled token embeddings.
  * "Ponder-and-refine" training: deep supervision (an LM head after every
    H-cycle) plus stochastic cycle depth, so the model learns *anytime*
    prediction and exposes a "think longer" knob at inference.

This is the single source of truth for the architecture, imported by both
``test.py`` (training) and ``temp.py`` (generation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


# ==========================================
# Configuration
# ==========================================
@dataclass
class HRMConfig:
    vocab_size: int
    n_embd: int = 256
    n_head: int = 4
    n_H_layers: int = 2          # blocks in the slow / high-level stack
    n_L_layers: int = 2          # blocks in the fast / low-level stack
    H_cycles: int = 2            # high-level cycles (== H_max for stochastic depth)
    L_cycles: int = 3            # low-level cycles per high-level cycle
    block_size: int = 128        # training context window (bounds the causal mask)
    embedding_scale: Optional[float] = None  # defaults to sqrt(n_embd)
    rope_theta: float = 10000.0
    dropout: float = 0.1

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.embedding_scale is None:
            self.embedding_scale = self.n_embd ** 0.5


# ==========================================
# Modern building blocks
# ==========================================
class RMSNorm(nn.Module):
    """Parameterless pre-RMSNorm."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


def _rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    """Return (cos, sin) each shaped (seq_len, head_dim) for rotary embeddings."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)            # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)     # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x, cos, sin):
    # x: (B, n_head, T, head_dim); cos/sin: (T, head_dim) -> broadcast over B, heads
    return x * cos + _rotate_half(x) * sin


def build_prefix_mask(token_type_ids):
    """Boolean attention mask for PrefixLM.

    token_type_ids: (B, T) with 1 == prefix, 0 == continuation.
    Returns (B, 1, T, T) where True means query i may attend to key j:

        allowed(i, j) = key_j_is_prefix OR (query_i_is_continuation AND j <= i)

    i.e. prefix tokens attend bidirectionally among themselves; continuation
    tokens attend causally to the prefix and to earlier continuation tokens.
    Both degenerate cases are handled: all-prefix -> fully bidirectional;
    all-continuation -> pure causal.
    """
    B, T = token_type_ids.shape
    device = token_type_ids.device
    tt = token_type_ids.bool()
    key_is_prefix = tt[:, None, :]                       # (B, 1, T) over key j
    query_is_cont = (~tt)[:, :, None]                    # (B, T, 1) over query i
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))[None]  # (1, T, T)
    allowed = key_is_prefix | (query_is_cont & causal)   # (B, T, T)
    return allowed[:, None, :, :]                        # (B, 1, T, T)


class GatedMHA(nn.Module):
    """Multi-head attention with RoPE and a sigmoid output gate.

    Causal by default; pass ``attn_mask`` for an explicit (PrefixLM) mask.
    """

    def __init__(self, cfg: HRMConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.rope_theta = cfg.rope_theta
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.gate = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = _rope_cache(T, self.head_dim, self.rope_theta, x.device, x.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=attn_mask is None,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(attn) * torch.sigmoid(self.gate(x))
        return self.resid_dropout(out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, cfg: HRMConfig):
        super().__init__()
        # ~8/3 * n_embd keeps params near a 4x ReLU FFN despite the extra gate proj.
        hidden = int(8 * cfg.n_embd / 3)
        hidden = (hidden + 7) // 8 * 8  # round up to a multiple of 8
        self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_value = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_out = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.w_out(F.silu(self.w_gate(x)) * self.w_value(x)))


class Block(nn.Module):
    """Pre-RMSNorm transformer block: gated attention then SwiGLU."""

    def __init__(self, cfg: HRMConfig):
        super().__init__()
        self.norm1 = RMSNorm()
        self.attn = GatedMHA(cfg)
        self.norm2 = RMSNorm()
        self.ffn = SwiGLU(cfg)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class Stack(nn.Module):
    """A stack of blocks (one timescale of the HRM core)."""

    def __init__(self, cfg: HRMConfig, n_layers: int):
        super().__init__()
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(n_layers)])

    def forward(self, x, attn_mask=None):
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        return x


# ==========================================
# HRM recurrent reasoning model
# ==========================================
class HRMModel(nn.Module):
    def __init__(self, cfg: HRMConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.H_module = Stack(cfg, cfg.n_H_layers)
        self.L_module = Stack(cfg, cfg.n_L_layers)
        # Learned seed for the fast (low-level) state.
        self.z_L_init = nn.Parameter(torch.zeros(1, 1, cfg.n_embd))
        self.ln_f = RMSNorm()
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # lecun-normal: unit variance after multiplying by embedding_scale.
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.n_embd ** -0.5)

    def _reason(self, idx, n_H_cycles, attn_mask):
        """Run the recurrent core, returning per-H-cycle logits (deep supervision)."""
        z_H = self.token_embedding(idx) * self.cfg.embedding_scale
        z_L = self.z_L_init.expand_as(z_H)
        cycle_logits = []
        for _ in range(n_H_cycles):
            for _ in range(self.cfg.L_cycles):
                z_L = self.L_module(z_L + z_H, attn_mask=attn_mask)
            z_H = self.H_module(z_H + z_L, attn_mask=attn_mask)
            cycle_logits.append(self.lm_head(self.ln_f(z_H)))
        return cycle_logits

    def forward(self, idx, targets=None, token_type_ids=None, n_H_cycles=None):
        n = n_H_cycles if n_H_cycles is not None else self.cfg.H_cycles
        # PrefixLM mask when token_type_ids is given; otherwise pure causal.
        attn_mask = build_prefix_mask(token_type_ids) if token_type_ids is not None else None
        cycle_logits = self._reason(idx, n, attn_mask)

        loss = None
        if targets is not None:
            # Under a bidirectional prefix, a position that predicts a prefix
            # token has already attended to it -> that prediction is leaky, so
            # we ignore those positions. A position predicts token i+1, which is
            # continuation iff token_type_ids[i+1] == 0 (the token after the
            # window is always treated as continuation).
            if token_type_ids is not None:
                target_types = torch.cat(
                    [token_type_ids[:, 1:], torch.zeros_like(token_type_ids[:, :1])],
                    dim=1,
                )
                targets = targets.masked_fill(target_types.bool(), -100)
            # Deep supervision: weighted mean of per-cycle cross-entropy, with
            # linearly increasing weight toward later (more refined) cycles.
            weights = torch.arange(1, n + 1, device=idx.device, dtype=torch.float)
            weights = weights / weights.sum()
            loss = idx.new_zeros((), dtype=torch.float)
            for w, logits in zip(weights, cycle_logits):
                step = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-100,
                )
                loss = loss + w * step
        return cycle_logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, n_cycles=None, temperature=1.0):
        """Autoregressively sample. ``n_cycles`` is the 'think longer' knob.

        The initial ``idx`` is treated as the bidirectional PrefixLM prefix;
        generated tokens are causal continuation (matching the HRM-Text
        inference contract). As the context window scrolls past the prompt the
        mask degrades gracefully to pure causal.
        """
        was_training = self.training
        self.eval()
        prompt_len = idx.size(1)
        for _ in range(max_new_tokens):
            cur_len = idx.size(1)
            start = max(0, cur_len - self.cfg.block_size)
            idx_cond = idx[:, start:]
            # Positions before prompt_len (in absolute terms) are prefix.
            abs_pos = torch.arange(start, cur_len, device=idx.device)
            ttids = (abs_pos < prompt_len).long().unsqueeze(0).expand(idx_cond.size(0), -1)
            cycle_logits, _ = self(idx_cond, token_type_ids=ttids, n_H_cycles=n_cycles)
            logits = cycle_logits[-1][:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        if was_training:
            self.train()
        return idx


def build_model(cfg: HRMConfig) -> HRMModel:
    """Construct an HRM model from a config (single construction entry point)."""
    return HRMModel(cfg)
