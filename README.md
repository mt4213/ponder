# ponder

Tiny char-level test model playing with the [HRM-Text](https://huggingface.co/sapientinc/HRM-Text-1B) ideas:
a two-speed recurrent core (slow `H` / fast `L` stacks), RoPE/SwiGLU/RMSNorm blocks,
PrefixLM masking, and a "think longer" knob (run more cycles = more compute).

Just an experiment, trained on `hamlet.txt`. Expect gibberish.

```bash
pip install -r requirements.txt
python train.py      # trains, writes ponder_checkpoint.pth
python generate.py   # generates at 1 vs N cycles
pytest -q            # tests
```

Apache 2.0.
