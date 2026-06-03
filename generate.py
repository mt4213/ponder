"""Load the trained HRM-style model and demonstrate the "think longer" knob.

Architecture is reconstructed from hrm_model.py using the config stored in the
checkpoint, then the same prompt is generated at different H-cycle counts so the
anytime / per-cycle refinement behaviour is visible.
"""
import torch

from hrm_model import HRMConfig, build_model

# ==========================================
# 1. Device + checkpoint
# ==========================================
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}\n")

checkpoint_path = "ponder_checkpoint.pth"
print(f"Loading checkpoint from '{checkpoint_path}'...")
checkpoint = torch.load(checkpoint_path, map_location=device)

# ==========================================
# 2. Rebuild tokenizer from checkpoint
# ==========================================
import tiktoken
tokenizer_name = checkpoint.get('tokenizer', 'cl100k_base')
enc = tiktoken.get_encoding(tokenizer_name)
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# ==========================================
# 3. Rebuild model from checkpoint config
# ==========================================
config = HRMConfig(
    vocab_size=checkpoint['vocab_size'],
    n_embd=checkpoint['n_embd'],
    n_head=checkpoint['n_head'],
    n_H_layers=checkpoint['n_H_layers'],
    n_L_layers=checkpoint['n_L_layers'],
    H_cycles=checkpoint['H_cycles'],
    L_cycles=checkpoint['L_cycles'],
    block_size=checkpoint['block_size'],
    embedding_scale=checkpoint['embedding_scale'],
    rope_theta=checkpoint['rope_theta'],
    dropout=0.0,  # no dropout at inference
)
model = build_model(config).to(device)
state_dict = checkpoint['model_state_dict']
uncompiled_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(uncompiled_state_dict)
model.eval()
print("Model and weights successfully loaded.\n")

# ==========================================
# 4. Generate at different reasoning depths
# ==========================================
input_prompt = "What is thy name?"

encoded_prompt = encode(input_prompt)

context = torch.tensor([encoded_prompt], dtype=torch.long, device=device)

print(f"Prompt: '{input_prompt}'")
print("(The prompt is the bidirectional PrefixLM block; continuation is causal.)")
print("Demonstrating the 'think longer' knob (more H-cycles = more reasoning):\n")

torch.manual_seed(0)
for n_cycles in (1, config.H_cycles):
    torch.manual_seed(0)  # same noise, so differences reflect reasoning depth
    generated = model.generate(
        context, max_new_tokens=500, n_cycles=n_cycles
    )[0].tolist()
    print("-" * 40)
    print(f"n_cycles = {n_cycles}")
    print("-" * 40)
    print(decode(generated))
    print()
