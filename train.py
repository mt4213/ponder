"""Train the HRM-style char-level reasoning model on hamlet.txt.

Architecture lives in hrm_model.py (shared with temp.py). Training uses the
"ponder-and-refine" recipe: deep supervision across H-cycles plus stochastic
cycle depth, so the model learns anytime prediction.
"""
import torch
from datasets import load_dataset

from hrm_model import HRMConfig, build_model

# ==========================================
# 1. Configuration
# ==========================================
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}\n")

torch.manual_seed(42)

# Training hyperparameters
block_size = 1024     # Context length
batch_size = 1        # Balanced batch size
grad_accum_steps = 16
learning_rate = 5e-4
max_iters = 5000
eval_interval = 500

# Model hyperparameters (HRM recurrent core + modern blocks)
n_embd = 512
n_head = 8
n_H_layers = 4        # slow / high-level stack
n_L_layers = 4        # fast / low-level stack
H_cycles = 2          # also the H_max for stochastic depth
L_cycles = 3
dropout = 0.1

# ==========================================
# 2. Dataset Setup
# ==========================================
import tiktoken
dataset = load_dataset("openai/gsm8k", "main")
text = ""
for item in dataset["train"]:
    text += f"Question: {item['question']}\nAnswer: {item['answer']}\n<|endoftext|>\n"

enc = tiktoken.get_encoding("cl100k_base")
vocab_size = enc.n_vocab
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

data = torch.tensor(encode(text), dtype=torch.long, device=device)


def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


# ==========================================
# 3. Build model
# ==========================================
config = HRMConfig(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_H_layers=n_H_layers,
    n_L_layers=n_L_layers,
    H_cycles=H_cycles,
    L_cycles=L_cycles,
    block_size=block_size,
    dropout=dropout,
)
model = build_model(config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params/1e6:.2f}M\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


def sample_prefix_mask(n_rows):
    """Random per-sequence PrefixLM split: positions < s are bidirectional
    prefix, the rest are causal continuation (UL2-style PrefixLM objective)."""
    s = torch.randint(1, block_size, (n_rows, 1), device=device)
    positions = torch.arange(block_size, device=device).unsqueeze(0)
    return (positions < s).long()  # (n_rows, block_size)


@torch.no_grad()
def final_cycle_loss(xb, yb):
    """Causal LM proxy: cross-entropy of the last (most refined) H-cycle at full
    depth, over all positions. A fixed yardstick for monitoring, independent of
    the sampled prefix split."""
    cycle_logits, _ = model(xb, n_H_cycles=H_cycles)
    logits = cycle_logits[-1]
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), yb.reshape(-1)
    ).item()


# ==========================================
# 4. Training (ponder-and-refine)
# ==========================================
print("Training HRM-style reasoning model...")
for iteration in range(max_iters):
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    
    for micro_step in range(grad_accum_steps):
        xb, yb = get_batch()

        # Stochastic depth: sample how many H-cycles to "think" this step.
        n_cycles = int(torch.randint(1, H_cycles + 1, (1,)).item())
        # PrefixLM: random per-sequence prefix/continuation split.
        token_type_ids = sample_prefix_mask(batch_size)
        _, loss = model(xb, yb, token_type_ids=token_type_ids, n_H_cycles=n_cycles)

        loss = loss / grad_accum_steps
        loss.backward()
        accum_loss += loss.item()

    optimizer.step()

    if iteration % eval_interval == 0 or iteration == max_iters - 1:
        fc = final_cycle_loss(xb, yb)
        print(
            f"Iteration {iteration:4d} | DeepSup Loss: {accum_loss:.4f} "
            f"| Final-cycle Loss: {fc:.4f} | sampled H-cycles: {n_cycles}"
        )

# ==========================================
# 5. Save the checkpoint
# ==========================================
checkpoint_path = "ponder_checkpoint.pth"
checkpoint = {
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'n_embd': n_embd,
    'n_head': n_head,
    'n_H_layers': n_H_layers,
    'n_L_layers': n_L_layers,
    'H_cycles': H_cycles,
    'L_cycles': L_cycles,
    'block_size': block_size,
    'embedding_scale': config.embedding_scale,
    'rope_theta': config.rope_theta,
    'tokenizer': 'cl100k_base',
}
torch.save(checkpoint, checkpoint_path)
print(f"\nModel saved to {checkpoint_path}")
