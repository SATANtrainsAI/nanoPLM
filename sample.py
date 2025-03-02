#!/usr/bin/env python3
"""
Batched inference for a protein GPT model with enhanced repetition control.
Processes all sample prefixes in a single batch and applies both a consecutive
repetition penalty and n-gram blocking to avoid repetitive patterns.

Note:
-----
This script loads a pre-trained GPT model and tokenizer definitions from an
external module (e.g., 'PLM.py'). Make sure the following references are
available in your environment:
    - GPT, GPT_Config, ProteinTokenizer, token_dict
    - adjust_logits_for_repetition, adjust_logits_for_ngram_blocking
"""


import argparse
import os
import torch
import torch.nn.functional as F
import math

# =============================================================================
# (Re)Use your model definitions: GPT, GPT_Config, ProteinTokenizer, token_dict, etc.
# from an external file (e.g., PLM.py)
# =============================================================================
from PLM import (
    GPT,
    GPT_Config,
    ProteinTokenizer,
    token_dict,
    adjust_logits_for_repetition,
    adjust_logits_for_ngram_blocking
)


# -----------------------------------------------------------------------------
# Sampling Functions
# -----------------------------------------------------------------------------
def sample_top_k(logits, top_k):
    """
    Sample the next token using top-k sampling.
      1. Select the top_k highest-scoring tokens in 'logits'.
      2. Convert these logits to probabilities via softmax.
      3. Sample a single token index from these top_k candidates.
    """
    top_logits, top_indices = torch.topk(logits, top_k, dim=-1)
    probs = F.softmax(top_logits, dim=-1)
    indices = torch.multinomial(probs, num_samples=1)
    next_tokens = top_indices.gather(-1, indices)
    return next_tokens

def sample_greedy(logits):
    """
    Sample the next token using greedy sampling.
    This simply takes the argmax of 'logits' along the last dimension.
    """
    return torch.argmax(logits, dim=-1, keepdim=True)

def sample_top_p(logits, top_p):
    """
    Sample the next token using top-p (nucleus) sampling.
      1. Compute the probability distribution via softmax.
      2. Sort tokens by their probability, then take a cumulative sum.
      3. Zero out probabilities once the sum exceeds top_p.
      4. Renormalize the remaining probabilities and sample from them.
    """
    batch_size = logits.size(0)
    next_tokens = []

    for i in range(batch_size):
        logits_i = logits[i]
        probs_i = F.softmax(logits_i, dim=-1)
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs_i, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find where cumulative_probs exceeds top_p
        mask = cumulative_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()

        # Sample a token from the "nucleus"
        token = torch.multinomial(sorted_probs, num_samples=1)
        token = sorted_indices[token]
        next_tokens.append(token)

    # Stack results into a tensor with shape (batch_size, 1)
    next_tokens = torch.stack(next_tokens, dim=0).unsqueeze(1)
    return next_tokens


# -----------------------------------------------------------------------------
# Batched Generation Function
# -----------------------------------------------------------------------------
def generate_batch(
    model,
    tokenizer,
    prefixes,
    max_sizes,
    sampling_method="top_k",
    sampling_args=None,
    rep_penalty=4,
    ngram_block=3
):
    """
    Generate sequences in batch for a list of prefixes using various sampling methods.

    Args:
        model (nn.Module):
            The GPT model (already loaded).
        tokenizer (ProteinTokenizer):
            The tokenizer instance (for encoding/decoding).
        prefixes (list of str):
            List of prefix strings to start generation from.
        max_sizes (list of int):
            List of maximum total lengths to generate for each prefix.
        sampling_method (str):
            One of "top_k", "greedy", or "top_p".
        sampling_args (dict, optional):
            Dictionary with keys like "temperature", "top_k", "top_p" to control sampling.
            Defaults to {"temperature": 1.0, "top_k": 7, "top_p": 0.5}.
        rep_penalty (int):
            Threshold for consecutive repetition penalty (blocks repeated tokens).
        ngram_block (int):
            N value for n-gram blocking (to block repeated n-grams).

    Returns:
        list of torch.Tensor:
            Each element is a single generated sequence tensor (shape [1, L_generated]).
    """
    # Default sampling arguments
    if sampling_args is None:
        sampling_args = {"temperature": 1.0, "top_k": 7, "top_p": 0.5}

    temperature = sampling_args.get("temperature", 1.0)
    top_k = sampling_args.get("top_k", 7)
    top_p = sampling_args.get("top_p", 0.5)

    device = next(model.parameters()).device

    # Encode each prefix into a tensor of token IDs. (Plus some example suffix?)
    # If you don't want to add [2, 30, 1, 16], you can remove that part.
    batch = [
        torch.tensor(tokenizer.encode(prefix) + [2, 30, 1, 16], dtype=torch.long, device=device).unsqueeze(0)
        for prefix in prefixes
    ]

    # Track finished sequences
    finished = [False] * len(batch)

    while not all(finished):
        # Pad each sequence in the batch to the same length for a single forward pass
        max_len = max(t.size(1) for t in batch)
        padded_batch = []
        for t in batch:
            pad_len = max_len - t.size(1)
            if pad_len > 0:
                t = F.pad(t, (0, pad_len), value=tokenizer.token_dict[tokenizer.pad_token])
            padded_batch.append(t)
        padded_batch = torch.cat(padded_batch, dim=0)  # shape (B, max_len)

        # Forward pass to get logits for the "last token" in each sequence
        # Many GPT implementations will return shape (B, 1, vocab_size) when targets=None.
        logits, _ = model(padded_batch, targets=None)
        next_logits = logits.squeeze(1)  # shape (B, vocab_size)

        # Apply repetition and n-gram penalties
        next_logits = adjust_logits_for_repetition(next_logits, padded_batch, rep_penalty)
        next_logits = adjust_logits_for_ngram_blocking(next_logits, padded_batch, n=ngram_block)

        # Now sample the next token from these adjusted logits
        if sampling_method == "top_k":
            next_tokens = sample_top_k(next_logits / temperature, top_k)
        elif sampling_method == "greedy":
            next_tokens = sample_greedy(next_logits)
        elif sampling_method == "top_p":
            next_tokens = sample_top_p(next_logits / temperature, top_p)
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")

        # Append the sampled token to each sequence if it's not finished
        new_batch = []
        for i, t in enumerate(batch):
            if finished[i]:
                new_batch.append(t)
                continue
            token = next_tokens[i]
            t = torch.cat([t, token.unsqueeze(0)], dim=1)
            new_batch.append(t)

            # Check stopping conditions: if <eos> is generated or max length reached
            if token.item() == tokenizer.eos_id or t.size(1) >= max_sizes[i]:
                finished[i] = True
        batch = new_batch

    return batch


# -----------------------------------------------------------------------------
# Main Inference Function
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Batched Inference for protein GPT model")
parser.add_argument(
    "--ckpt", type=str, default="/content/drive/MyDrive/Language Model/ckpt_ppi.pt",
    help="Path to the pretrained model checkpoint"
)
parser.add_argument(
    "--sampling_method", type=str, choices=["top_k", "greedy", "top_p"],
    default="top_k", help="Sampling method to use"
)
parser.add_argument(
    "--temperature", type=float, default=0.7,
    help="Sampling temperature"
)
parser.add_argument(
    "--top_k", type=int, default=4,
    help="Top-k value for top-k sampling"
)
parser.add_argument(
    "--top_p", type=float, default=0.5,
    help="Top-p threshold for top-p (nucleus) sampling"
)
parser.add_argument(
    "--num_samples", type=int, default=8,
    help="Number of sequences to generate per prefix"
)
parser.add_argument(
    "--ngram", type=int, default=4,
    help="N-gram size for blocking repetitive patterns"
)
args = parser.parse_args('')  # For interactive testing, you can override with a custom arg string.

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# Load Model Checkpoint
# -----------------------------------------------------------------------------
checkpoint = torch.load(args.ckpt, map_location=device)
if "model_args" not in checkpoint:
    raise ValueError("Checkpoint does not contain model_args")
model_args = checkpoint["model_args"]

# Instantiate the GPT model config and create the model
cfg = GPT_Config(**model_args)
model = GPT(cfg)
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

# Create tokenizer
tokenizer = ProteinTokenizer(token_dict)

# Sampling arguments (temperature, top_k, top_p, etc.)
sampling_args = {
    "temperature": args.temperature,
    "top_k": args.top_k,
    "top_p": args.top_p
}

# -----------------------------------------------------------------------------
# Sample Generation
# -----------------------------------------------------------------------------
# Example list of prefix strings (just 1 long prefix here) and their maximum lengths
sample_prefixes = [
    "MDSSSTEQTVKQKLRRVIFGTDTKAGRYFDISLIICIILSVLLVFIDTVDSVHKEYGGVIRIVEWVFTGIFTLEYLLRLYCSAQPVQYARSFYGIVDLLSILPSYLALIFPGANFTLVIRILRLFRIFRVLKLLRYLSEGNILLRAMMQSSRKVFLFFFSVSLIVMVLSAFMYVVEGPENGFTSIPKSIYWTIVTITTVGYGDITPQTALGQGIAALTMLIGYSIIAIPTGILTAEISQEIVRKKDLRRCSNCLKTGHEINALYCDKCGSELESDL"
]
sizes = [800]

# We replicate each prefix multiple times for multi-sample generation
batched_prefixes = []
batched_sizes = []
for prefix, size in zip(sample_prefixes, sizes):
    for _ in range(args.num_samples):
        batched_prefixes.append(prefix)
        batched_sizes.append(size)

# Perform batched generation
generated_batch = generate_batch(
    model,
    tokenizer,
    batched_prefixes,
    batched_sizes,
    sampling_method=args.sampling_method,
    sampling_args=sampling_args,
    rep_penalty=4,
    ngram_block=args.ngram
)

# -----------------------------------------------------------------------------
# Print Results
# -----------------------------------------------------------------------------
num_prefixes = len(sample_prefixes)
for i in range(num_prefixes):
    print(f"Generating samples for prefix: {sample_prefixes[i]}")
    for j in range(args.num_samples):
        idx = i * args.num_samples + j
        gen_tensor = generated_batch[idx]
        # gen_tensor is shape [1, L], we decode the single row
        generated_text = tokenizer.decode(gen_tensor[0].tolist())
        print(f"len: {len(generated_text)} - Sample {j+1}:\n{generated_text}\n{'-'*80}")
