"""Evaluate the HRM model on the GSM8K test set."""
import torch
from datasets import load_dataset
import tiktoken
import re
from hrm_model import HRMConfig, build_model

# ==========================================
# 1. Configuration
# ==========================================
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Use the fine-tuned checkpoint if it exists, otherwise fall back to the pre-trained one
import os
checkpoint_path = "ponder_finetuned.pth" if os.path.exists("ponder_finetuned.pth") else "ponder_checkpoint.pth"
print(f"Loading checkpoint from '{checkpoint_path}'...")
checkpoint = torch.load(checkpoint_path, map_location=device)

import tiktoken
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
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded.\n")

# ==========================================
# 2. Evaluation Logic
# ==========================================
def extract_answer(text):
    """Extracts the final numerical answer from a GSM8K solution string."""
    # The true answer is typically right after ####
    if "####" in text:
        ans = text.split("####")[-1].strip()
        # strip out any commas from numbers
        ans = ans.replace(",", "")
        return ans
    
    # If the model didn't use ####, try to find the last number in the string
    numbers = re.findall(r'-?\d*\.?\d+', text.replace(",", ""))
    if numbers:
        return numbers[-1]
    return None

print("Loading GSM8K test set...")
dataset = load_dataset("openai/gsm8k", "main", split="test")

# Let's test on the first 50 examples to save time, you can change this!
num_to_eval = 50
correct = 0
total = 0

print(f"Evaluating on {num_to_eval} examples...")

with torch.no_grad():
    for i in range(num_to_eval):
        item = dataset[i]
        question = item['question']
        ground_truth = extract_answer(item['answer'])
        
        input_prompt = f"Question: {question}\nAnswer:"
        encoded_prompt = encode(input_prompt)
        context = torch.tensor([encoded_prompt], dtype=torch.long, device=device)
        
        # Generate the response (using max reasoning cycles!)
        generated_tokens = model.generate(
            context, 
            max_new_tokens=150, 
            n_cycles=config.H_cycles
        )[0].tolist()
        
        output_text = decode(generated_tokens)
        
        # Isolate the model's generated answer text
        model_answer_text = output_text[len(input_prompt):]
        model_final_number = extract_answer(model_answer_text)
        
        if model_final_number == ground_truth:
            correct += 1
            print(f"[{i+1}/{num_to_eval}] Correct! (Ans: {ground_truth})")
        else:
            print(f"[{i+1}/{num_to_eval}] Incorrect. Expected: {ground_truth}, Got: {model_final_number}")
            
        total += 1

accuracy = (correct / total) * 100
print(f"\nFinal Accuracy on subset: {accuracy:.2f}% ({correct}/{total})")
