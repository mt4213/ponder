"""Interactive CLI to chat with the trained HRM model."""
import torch
import sys
import os
import tiktoken
from hrm_model import HRMConfig, build_model

# ==========================================
# 1. Device + Checkpoint
# ==========================================
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# Auto-detect if we have a fine-tuned model or just the pre-trained base
checkpoint_path = "ponder_finetuned.pth" if os.path.exists("ponder_finetuned.pth") else "ponder_checkpoint.pth"

print(f"\n[System] Loading checkpoint from '{checkpoint_path}' onto {device.upper()}...")
checkpoint = torch.load(checkpoint_path, map_location=device)

# ==========================================
# 2. Tokenizer & Config Setup
# ==========================================
tokenizer_name = checkpoint.get('tokenizer', 'cl100k_base')
enc = tiktoken.get_encoding(tokenizer_name)
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

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
    dropout=0.0,
)

model = build_model(config).to(device)

state_dict = checkpoint['model_state_dict']
uncompiled_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(uncompiled_state_dict)

model.eval()
print("[System] Model and weights successfully loaded.\n")
print("="*60)
print("Type your message and press ENTER to talk to the model.")
print("Type 'quit' or 'exit' to stop.")
print("="*60 + "\n")

# ==========================================
# 3. Interactive Loop
# ==========================================
while True:
    try:
        user_input = input("\nYou: ")
        if user_input.strip().lower() in ['quit', 'exit']:
            print("Exiting...")
            break
            
        if not user_input.strip():
            continue
            
        # Add a trailing newline to simulate continuation
        prompt = user_input + "\n"
        encoded_prompt = encode(prompt)
        context = torch.tensor([encoded_prompt], dtype=torch.long, device=device)
        
        # Generate the response using maximum reasoning cycles
        with torch.no_grad():
            generated_tokens = model.generate(
                context, 
                max_new_tokens=200, 
                n_cycles=config.H_cycles
            )[0].tolist()
            
        # The model returns the prompt + generated tokens. 
        # We strip out the prompt part so it only prints the response.
        output_text = decode(generated_tokens[len(encoded_prompt):])
        
        print(f"\nModel: {output_text.strip()}")
        
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"\nAn error occurred: {e}")
