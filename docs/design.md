# Design: HRM-style char-level reasoning model

Date: 2026-06-03
Status: Approved (brainstorming) — ready for implementation plan
Location: this repository (docs/design.md)

## Goal

Rewrite the from-scratch char-level transformer in `.antigravitycli/test.py` to adopt
the architecture of [`sapientinc/HRM-Text-1B`](https://huggingface.co/sapientinc/HRM-Text-1B):
a **dual-timescale recurrent reasoning core** plus modern transformer components,
keeping the existing char-level training on `hamlet.txt`.

This is NOT loading the real 1B checkpoint — it is a faithful, tiny, from-scratch
re-implementation of HRM's *ideas*, trainable on CPU/single GPU.

## Decisions (locked during brainstorming)

1. **Scope:** Full HRM core + modern blocks (not just modern blocks, not the real checkpoint).
2. **Training/gradient strategy:** Approach A — "ponder-and-refine" (deep supervision +
   stochastic cycle depth + full BPTT). Deliberately departs from HRM-Text-1B's
   conservative single-final-loss / fixed-cycle recipe to expose observable reasoning behavior.
3. **Inference scripts:** Update `temp.py` to match the new architecture and checkpoint format.
4. **Sizing default:** 2-layer H stack, 2-layer L stack, `H_cycles=2`, `L_cycles=3`
   (HRM's 2x3), keeping params near the current 4-block budget.

## 1. File layout (fixes existing duplication)

Today `test.py` and `temp.py` each define their own copy of the model and drift apart.
Extract the architecture into a single source of truth:

- **`hrm_model.py`** — `HRMConfig` dataclass + all model modules + `build_model(config)`.
- **`test.py`** — data loading, training loop, checkpoint save. Imports from `hrm_model`.
- **`temp.py`** — checkpoint load + generation demo. Imports from `hrm_model`.

All three remain plain scripts run directly; only the duplicated model is removed.

## 2. Modern transformer block (shared by H and L stacks)

Replaces the current block internals (LayerNorm/ReLU/learned-pos):

- **RMSNorm**, parameterless, pre-norm.
- **RoPE** rotary positional embeddings (theta=10000) instead of a learned
  `position_embedding` table — removes the fixed positional table; block_size only
  bounds the causal mask / training window.
- **SwiGLU** FFN instead of `Linear -> ReLU -> Linear`.
- **Gated MHA**: standard causal multi-head attention with a sigmoid output gate.
- **Scaled token embedding** with lecun-normal init.

Dropout retained (default 0.1 in training, 0.0 in inference) on attention weights and
FFN/projection outputs.

## 3. HRM recurrent reasoning core

Two separate stacks: `H_module` (slow) and `L_module` (fast), each a few modern blocks.
A learned `z_L_init` parameter seeds the fast state. Forward pass (training, with deep
supervision):

```
z_H = embed(idx) * embedding_scale          # (B, T, C)
z_L = z_L_init.expand_as(z_H)
cycle_logits = []
for h in range(n_H_cycles):                 # default 2 (or sampled during training)
    for l in range(L_cycles):               # default 3
        z_L = L_module(z_L + z_H)
    z_H = H_module(z_H + z_L)
    cycle_logits.append(lm_head(ln_f(z_H))) # one LM head output per H-cycle
return cycle_logits                          # list length == n_H_cycles
```

- Attention mask is **causal by default**, or a **PrefixLM mask** when
  `token_type_ids` is supplied (see section 4a). Same mask is applied in every block
  of both H and L stacks, every cycle.
- `lm_head`, `ln_f` (final RMSNorm), and the embedding are shared across cycles.
- Param budget kept near current model by sizing H/L stacks to ~2 layers each.

## 4. Ponder-and-refine training (Approach A)

- **Deep supervision:** total loss = weighted mean of per-cycle cross-entropy over all
  H-cycle logits. Weights linearly increase with cycle index (default), so cycle 1
  remains a valid fast guess and the final cycle is the refined answer. Weights normalized
  to sum to 1 so loss magnitude is comparable across sampled depths.
- **Stochastic depth:** each training step samples `n_H_cycles ~ Uniform{1..H_max}`
  (`H_max` default 2; `L_cycles` fixed). Trains anytime prediction.
- **Full BPTT** through the sampled cycles (memory is a non-issue at this scale).
- Optimizer/data unchanged in spirit: AdamW, char-level `hamlet.txt`, comparable iters.
  Reported training loss is the deep-supervision total; also print the final-cycle loss
  for interpretability.

## 4a. PrefixLM masking (added after initial approval)

HRM-Text is a prefix-LM: prompt tokens attend bidirectionally, continuation tokens
causally. To reproduce this faithfully on flat text (no prompt/response boundary) we
use the UL2-style PrefixLM objective with a **random per-sequence split**:

- Per training window sample a split `s ~ Uniform{1..block_size-1}`. Positions `[0,s)`
  are bidirectional prefix, `[s,T)` are causal continuation, encoded as
  `token_type_ids` (1 = prefix, 0 = continuation).
- Mask rule, `allowed(i,j) = key_j_is_prefix OR (query_i_is_continuation AND j<=i)`.
  Degenerates correctly: all-prefix -> fully bidirectional; all-continuation -> causal.
- **Loss masks prefix-target positions:** a position predicting a prefix token has
  already attended to it (leaky), so those positions use `ignore_index`. Only positions
  whose *target* is continuation contribute (the token past the window counts as
  continuation, so every sequence always has >=1 supervised position).
- Deep supervision (every cycle) and stochastic depth compose unchanged with this.

At inference `generate()` marks the initial prompt as the bidirectional prefix and
generated tokens as causal continuation; as the context window scrolls past the prompt
the mask degrades gracefully to pure causal.

## 5. Inference: the "think longer" knob

`generate(idx, max_new_tokens, n_cycles=None)` runs the core at a chosen H-cycle count
(defaults to the trained `H_max`). The last cycle's logits drive sampling.

`temp.py` demonstrates the anytime property: generate the same prompt at `n_cycles=1`
versus `n_cycles=H_max` (and optionally more) so per-cycle refinement is visible.

## 6. Checkpoint format

Extend the current saved dict with the full `HRMConfig` so `temp.py` rebuilds the exact
model:

- existing: `model_state_dict`, `vocab_size`, `block_size`, `char_to_ix`, `ix_to_char`
- new: `n_embd`, `n_head`, `n_H_layers`, `n_L_layers`, `H_cycles` (== H_max), `L_cycles`,
  `embedding_scale`, `rope_theta`, `dropout` (stored as 0.0 for inference rebuild).

The old `n_layer` field is removed (replaced by `n_H_layers` / `n_L_layers`); old
checkpoints are not loadable by the new `temp.py` (acceptable — toy/scratch).

## 7. Verification

- `hrm_model.py` imports clean.
- Smoke test on CPU with a tiny config (small vocab, n_embd, T):
  - forward returns a list of per-cycle logits with shape `(B, T, vocab)`;
  - loss path runs and is finite;
  - `generate` runs at `n_cycles=1` and `n_cycles=H_max` and returns the right length.
- `test.py`: train a handful of iters; confirm loss decreases and checkpoint saves.
- `temp.py`: load checkpoint and generate at 1 vs N cycles without error.

## Non-goals / YAGNI

- No learned halting / ACT (Approach B) — explicitly deferred.
- No one-step/DEQ fixed-point gradient (Approach C).
- No loading the real 1B checkpoint.
- No subword tokenizer; stays char-level.

(Note: PrefixLM masking was initially a non-goal but was added after approval — see
section 4a.)
