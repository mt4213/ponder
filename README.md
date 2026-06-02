# ponder

A tiny, from-scratch, char-level re-implementation of the architectural ideas behind
[`sapientinc/HRM-Text-1B`](https://huggingface.co/sapientinc/HRM-Text-1B) — the
**Hierarchical Reasoning Model (HRM)** — trainable on a single GPU (or CPU) on a small
text file.

This is *not* the 1B checkpoint. It is a small model that reproduces HRM's ideas so they
can be read, trained, and poked at in a couple hundred lines.

## What makes it an HRM (and not just a small GPT)

- **Dual-timescale recurrent reasoning core.** Two transformer stacks — a slow
  high-level module `H` and a fast low-level module `L` — iterate over the *same*
  embeddings with additive state injection (`z_L + z_H`):

  ```
  z_H = embed(idx) * scale
  z_L = z_L_init
  for h in range(H_cycles):        # default 2
      for l in range(L_cycles):    # default 3
          z_L = L(z_L + z_H)
      z_H = H(z_H + z_L)
  ```

  This buys extra effective compute depth at a fixed parameter count.

- **Modern blocks:** RMSNorm (parameterless, pre-norm), RoPE, SwiGLU, gated multi-head
  attention, scaled token embeddings.

- **PrefixLM masking** (like HRM-Text): prompt tokens attend bidirectionally, generated
  tokens attend causally. Trained with the UL2-style objective (random per-sequence
  prefix/continuation split), so the model genuinely *is* a prefix-LM rather than
  matching the contract only at inference.

## The twist: "ponder-and-refine" training

Instead of HRM-Text's single final loss over a fixed cycle count, this repo trains with:

- **Deep supervision** — an LM head after *every* H-cycle, loss = weighted mean over
  cycles (later cycles weighted higher).
- **Stochastic cycle depth** — the number of H-cycles is sampled per step.

Together these teach *anytime* prediction and expose a **"think longer" knob**: run more
H-cycles at inference to spend more compute and refine the output.

```python
model.generate(ctx, max_new_tokens=500, n_cycles=1)   # fast guess
model.generate(ctx, max_new_tokens=500, n_cycles=2)   # think longer
```

## Usage

```bash
pip install -r requirements.txt

# Train on hamlet.txt -> writes ponder_checkpoint.pth
python train.py

# Reload and generate, showing the think-longer knob at 1 vs N cycles
python generate.py

# Tests
pytest -q
```

Run the scripts from the repo root (they read `./hamlet.txt` and write the checkpoint to
the current directory). `hamlet.txt` (public domain) is included as a tiny demo corpus —
swap in any larger plaintext file for better samples.

## Files

| File | Purpose |
|------|---------|
| `hrm_model.py` | Architecture: config, modern blocks, HRM recurrent core, PrefixLM mask |
| `train.py` | Training loop (ponder-and-refine + PrefixLM), checkpoint save |
| `generate.py` | Load checkpoint, generate, demonstrate the think-longer knob |
| `test_hrm_model.py` | Smoke tests |
| `docs/design.md` | Design rationale and decisions |

## Caveats

- It's tiny and trains on a tiny corpus — expect Shakespeare-flavored gibberish that
  sharpens with more data and iterations.
- Not instruction-tuned, not a chat model. It's a study of the architecture.

## License

[Apache 2.0](LICENSE).

HRM is the work of Sapient Inc.; this is an independent educational re-implementation of
the ideas, not affiliated with or derived from their weights.
