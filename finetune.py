"""Fine-tune the HRM model on terminal-bench trajectories."""
import torch
from datasets import load_dataset
import tiktoken
from hrm_model import HRMConfig, build_model

# ==========================================
# 1. Configuration
# ==========================================
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}\n")

torch.manual_seed(42)
if device == 'cuda':
    torch.set_float32_matmul_precision('high')

# Fine-tuning hyperparameters
block_size = 1024
batch_size = 4
grad_accum_steps = 4
learning_rate = 1e-5  # Much lower than pre-training!
max_iters = 2000
eval_interval = 200

# ==========================================
# 2. Dataset Setup
# ==========================================
print("Loading terminalbench trajectories...")
# Load the dataset (streaming to avoid downloading everything at once)
dataset = load_dataset("yoonholee/terminalbench-trajectories", split="train", streaming=True)

# Filter for successful trajectories (assuming reward == 1.0 means success)
successful_runs = dataset.filter(lambda x: x["reward"] == 1.0)

text = ""
print("Extracting text from successful trajectories...")
# We take the first 100 successful runs for fine-tuning
for i, item in enumerate(successful_runs):
    if i >= 100:
        break
    text += item["steps"] + "\n<|endoftext|>\n"

print(f"Extracted {len(text)} characters of high-quality trajectory data.")

enc = tiktoken.get_encoding("cl100k_base")
vocab_size = enc.n_vocab
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
data = torch.tensor(encode(text), dtype=torch.long, device=device)

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

# ==========================================
# 3. Build & Load Pre-trained Model
# ==========================================
print("Loading pre-trained checkpoint...")
checkpoint_path = "ponder_checkpoint.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

config = HRMConfig(
    vocab_size=vocab_size,
    n_embd=checkpoint['n_embd'],
    n_head=checkpoint['n_head'],
    n_H_layers=checkpoint['n_H_layers'],
    n_L_layers=checkpoint['n_L_layers'],
    H_cycles=checkpoint['H_cycles'],
    L_cycles=checkpoint['L_cycles'],
    block_size=block_size,
    dropout=0.1,
)

model = build_model(config).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

if device == 'cuda':
    model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def sample_prefix_mask(n_rows):
    s = torch.randint(1, block_size, (n_rows, 1), device=device)
    positions = torch.arange(block_size, device=device).unsqueeze(0)
    return (positions < s).long()

@torch.no_grad()
def final_cycle_loss(xb, yb):
    cycle_logits, _ = model(xb, n_H_cycles=config.H_cycles)
    logits = cycle_logits[-1]
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), yb.reshape(-1)
    ).item()

# ==========================================
# 4. Fine-tuning Loop
# ==========================================
print("Starting fine-tuning...")
for iteration in range(max_iters):
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    
    for micro_step in range(grad_accum_steps):
        xb, yb = get_batch()
        n_cycles = int(torch.randint(1, config.H_cycles + 1, (1,)).item())
        token_type_ids = sample_prefix_mask(batch_size)
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == 'cuda' else torch.float16):
            _, loss = model(xb, yb, token_type_ids=token_type_ids, n_H_cycles=n_cycles)

        loss = loss / grad_accum_steps
        loss.backward()
        accum_loss += loss.item()

    optimizer.step()

    if iteration % eval_interval == 0 or iteration == max_iters - 1:
        fc = final_cycle_loss(xb, yb)
        print(f"Iteration {iteration:4d} | DeepSup Loss: {accum_loss:.4f} | Final-cycle Loss: {fc:.4f}")

# ==========================================
# 5. Save Fine-tuned Checkpoint
# ==========================================
finetuned_path = "ponder_finetuned.pth"
checkpoint['model_state_dict'] = model.state_dict()
torch.save(checkpoint, finetuned_path)
print(f"\nFine-tuned model saved to {finetuned_path}")
