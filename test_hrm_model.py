"""Smoke tests for the HRM-style char-level model (hrm_model.py).

Run from this directory with the dyad venv:
    ../.venv/bin/python -m pytest test_hrm_model.py -q
"""
import torch

from hrm_model import HRMConfig, build_model


def tiny_config():
    return HRMConfig(
        vocab_size=17,
        n_embd=32,
        n_head=4,
        n_H_layers=1,
        n_L_layers=1,
        H_cycles=2,
        L_cycles=3,
        block_size=16,
        dropout=0.0,
    )


def test_forward_returns_per_cycle_logits():
    cfg = tiny_config()
    model = build_model(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))

    cycle_logits, loss = model(idx)

    # Deep supervision: one logits tensor per H-cycle.
    assert isinstance(cycle_logits, list)
    assert len(cycle_logits) == cfg.H_cycles
    for logits in cycle_logits:
        assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is None  # no targets provided


def test_loss_is_finite_with_targets():
    cfg = tiny_config()
    model = build_model(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))

    _, loss = model(idx, targets)

    assert loss is not None
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_stochastic_depth_override():
    cfg = tiny_config()
    model = build_model(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))

    cycle_logits, _ = model(idx, n_H_cycles=1)

    # Asked for a single H-cycle -> a single supervised head.
    assert len(cycle_logits) == 1


def test_generate_think_longer_knob():
    cfg = tiny_config()
    model = build_model(cfg)
    idx = torch.zeros((1, 1), dtype=torch.long)

    out_fast = model.generate(idx, max_new_tokens=5, n_cycles=1)
    out_slow = model.generate(idx, max_new_tokens=5, n_cycles=cfg.H_cycles)

    assert out_fast.shape == (1, 6)
    assert out_slow.shape == (1, 6)
    # values are valid token ids
    assert int(out_slow.max()) < cfg.vocab_size
    assert int(out_slow.min()) >= 0


def test_token_type_ids_enable_bidirectional_prefix():
    # With the whole sequence marked as prefix, attention is bidirectional, so
    # an early position's logits must depend on a *later* token.
    cfg = tiny_config()
    torch.manual_seed(0)
    model = build_model(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 6))
    ttids = torch.ones_like(idx)  # entire input is one bidirectional prefix block

    with torch.no_grad():
        base = model(idx, token_type_ids=ttids)[0][-1]
        idx2 = idx.clone()
        idx2[0, -1] = (idx[0, -1] + 1) % cfg.vocab_size
        changed = model(idx2, token_type_ids=ttids)[0][-1]

    assert not torch.allclose(base[0, 0], changed[0, 0])


def test_default_is_causal_earlier_independent_of_later():
    # No token_type_ids -> pure causal: an early position is unaffected by a
    # later token. Guards the default path.
    cfg = tiny_config()
    model = build_model(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 6))

    with torch.no_grad():
        base = model(idx)[0][-1]
        idx2 = idx.clone()
        idx2[0, -1] = (idx[0, -1] + 1) % cfg.vocab_size
        changed = model(idx2)[0][-1]

    assert torch.allclose(base[0, 0], changed[0, 0], atol=1e-6)


def test_prefix_loss_ignores_prefix_target_positions():
    # All-prefix token_type_ids -> only the final position predicts a
    # continuation target; loss must ignore prefix-target positions.
    cfg = tiny_config()
    model = build_model(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 6))
    ttids = torch.ones_like(idx)
    targets = torch.randint(0, cfg.vocab_size, (1, 6))

    with torch.no_grad():
        base = model(idx, targets, token_type_ids=ttids)[1]
        t_early = targets.clone()
        t_early[0, 0] = (targets[0, 0] + 1) % cfg.vocab_size
        same = model(idx, t_early, token_type_ids=ttids)[1]
        t_last = targets.clone()
        t_last[0, -1] = (targets[0, -1] + 1) % cfg.vocab_size
        diff = model(idx, t_last, token_type_ids=ttids)[1]

    assert torch.allclose(base, same)       # ignored position -> no change
    assert not torch.allclose(base, diff)   # counted position -> changes loss


def test_no_learned_position_table():
    # RoPE means there is no nn.Embedding for positions, so sequences longer
    # than block_size still run (only the causal window is bounded by block_size
    # at training time; generate crops to block_size internally).
    cfg = tiny_config()
    model = build_model(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    cycle_logits, _ = model(idx)
    assert cycle_logits[-1].shape == (1, cfg.block_size, cfg.vocab_size)
