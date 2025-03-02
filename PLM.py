"""
Credit to https://github.com/karpathy/nanoGPT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# We use IterableDataset to stream large datasets from disk
# and DDP for multi-GPU distributed training
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DistributedSampler, DataLoader

import argparse
import os
import time
from contextlib import nullcontext
from functools import partial
import random
import math
import inspect
from dataclasses import dataclass


# -----------------------------------------------------------------------------
# Utility Functions and Global Dicts
# -----------------------------------------------------------------------------

def exists(val):
    """Check if a value is not None."""
    return val is not None

def print0(*args, **kwargs):
    """
    Print only on the primary (rank=0) process in a distributed run.
    If not distributed, it prints normally.
    """
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


# -----------------------------------------------------------------------------
# Token Dictionaries
# - token_dict: maps tokens (e.g., A, T, etc.) to integer IDs
# - token_dict_inv: inverse mapping from IDs back to tokens
# -----------------------------------------------------------------------------
token_dict = {
    "<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "A": 4, "B": 5, "C": 6,
    "D": 7, "E": 8, "F": 9, "G": 10, "H": 11, "I": 12, "J": 13, "K": 14,
    "L": 15, "M": 16, "N": 17, "O": 18, "P": 19, "Q": 20, "R": 21, "S": 22,
    "T": 23, "U": 24, "V": 25, "W": 26, "X": 27, "Y": 28, "Z": 29, "1": 30,
    "2": 31
}
token_dict_inv = {v: k for k, v in token_dict.items()}


# -----------------------------------------------------------------------------
# ProteinTokenizer
# -----------------------------------------------------------------------------
class ProteinTokenizer:
    """
    Converts raw protein sequences (strings) into lists of token IDs, and vice versa.
    This tokenizer can also add special tokens <bos>, <eos> for sequence boundaries,
    and it provides a way to pad sequences to a fixed length.
    """
    def __init__(self, token_dict):
        self.token_dict = token_dict
        self.inv_token_dict = {v: k for k, v in token_dict.items()}
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.pad_id = token_dict[self.pad_token]
        self.bos_id = token_dict[self.bos_token]
        self.eos_id = token_dict[self.eos_token]
        self.stop_tokens = [token_dict[self.eos_token]]

    def tokenize(self, sequence):
        """
        Splits a sequence string into individual characters (tokens).
        e.g., 'ABC' -> ['A', 'B', 'C'].
        """
        return list(sequence)

    def convert_tokens_to_ids(self, tokens):
        """
        Converts a list of tokens (e.g., ['A','B','C']) into their corresponding IDs,
        using the provided token_dict. Unknown tokens default to <unk>.
        """
        return [
            self.token_dict.get(token, self.token_dict[self.unk_token]) for token in tokens
        ]

    def encode(self, sequence, add_special_tokens=True):
        """
        Goes from raw sequence string -> tokens -> token IDs.
        Optionally adds a <bos> token ID at the start.
        """
        tokens = self.tokenize(sequence)
        if add_special_tokens:
            tokens = [self.bos_token] + tokens
        return self.convert_tokens_to_ids(tokens)

    def decode(self, token_ids):
        """
        Converts token IDs back to a string representation (the inverse of `encode`).
        """
        tokens = [self.inv_token_dict[token_id] for token_id in token_ids]
        return "".join(tokens)

    def pad_sequences(self, sequences, padding_value=None, block_size=None):
        """
        Pads a list of sequences (each a list of token IDs) to a fixed block_size.
        Any extra space is filled with padding_value (default = <pad> ID).
        """
        if block_size is None:
            block_size = max(len(seq) for seq in sequences)
        if padding_value is None:
            padding_value = self.token_dict[self.pad_token]

        padded = []
        for seq in sequences:
            seq = list(seq)[:block_size]  # truncate if longer than block_size
            padding_needed = max(0, block_size - len(seq))
            seq += [padding_value] * padding_needed
            padded.append(seq)

        return padded


# -----------------------------------------------------------------------------
# Custom IterableDataset for Large FASTA Datasets
# -----------------------------------------------------------------------------
class ProteinIterableDataset(IterableDataset):
    """
    Reads protein sequences from a FASTA file line by line, yielding tokenized samples.
    The dataset can be split into 'train' or 'val' sets by specifying val_fraction.
    Additionally, this class allows for distributed reading by skipping sequences
    not intended for the current rank in multi-GPU training.
    """
    def __init__(
        self,
        fasta_file,
        tokenizer,
        block_size=1024,
        shuffle_buffer_size=10000,
        split='train',
        val_fraction=0.0001,
        rank=0,
        world_size=1
    ):
        self.fasta_file = fasta_file
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.split = split
        self.val_fraction = val_fraction
        self.rank = rank
        self.world_size = world_size

        # We define an approximate total number of sequences in the FASTA file.
        # For extremely large datasets, this is set manually rather than scanning.
        self.total_sequences = 240_000_000  # Adjust for your dataset

        # Decide how many sequences are for validation
        self.val_threshold = int(self.total_sequences * self.val_fraction)

        # seq_idx will keep track of which sequence number we are on
        self.seq_idx = -1

    def __iter__(self):
        """
        Streams the FASTA file line by line. For each sequence, we tokenize and
        possibly shuffle them in a buffer for randomness, then yield.
        """
        buffer = []
        with open(self.fasta_file, 'r') as f:
            seq = ''
            header = ''
            self.seq_idx = -1
            for line in f:
                line = line.strip()
                # New sequence found if the line starts with '>'
                if line.startswith('>'):
                    self.seq_idx += 1
                    if seq:
                        # Decide if this sequence belongs to train or val split
                        if self._is_in_split(self.seq_idx):
                            # Skip if not assigned to this rank in distributed training
                            if self.split == 'train' and self.seq_idx % self.world_size != self.rank:
                                seq = ''
                                continue
                            # Add seq to buffer
                            buffer.append(seq)
                            # If our buffer is big enough, shuffle and yield
                            if len(buffer) >= self.shuffle_buffer_size:
                                random.shuffle(buffer)
                                for buffered_seq in buffer:
                                    yield self.process_sequence(buffered_seq)
                                buffer = []
                        seq = ''
                    header = line
                else:
                    # If not a header, we append the sequence content
                    seq += line.strip().upper()

            # End of file: process last sequence if it exists
            if seq:
                if self._is_in_split(self.seq_idx):
                    if self.split == 'train' and self.seq_idx % self.world_size != self.rank:
                        seq = ''
                    else:
                        buffer.append(seq)

            # Yield any leftover buffer after the file ends
            if buffer:
                random.shuffle(buffer)
                for buffered_seq in buffer:
                    yield self.process_sequence(buffered_seq)

    def __len__(self):
        """
        Returns an approximate size: total sequences in 'train' or 'val',
        divided by the number of distributed processes.
        """
        if self.split == 'train':
            train_sequences = self.total_sequences - self.val_threshold
            return train_sequences // self.world_size
        elif self.split == 'val':
            return self.val_threshold // self.world_size
        return 0

    def _is_in_split(self, seq_idx):
        """Check if a given sequence index belongs to train or val split."""
        if self.split == 'train':
            return seq_idx >= self.val_threshold
        elif self.split == 'val':
            return seq_idx < self.val_threshold
        return False

    def process_sequence(self, sequence):
        """
        Apply final processing: filter invalid chars, encode, pad/truncate
        for block_size, then produce (inputs, targets) for language modeling.
        """
        valid_chars = set(self.tokenizer.token_dict.keys()) - {
            self.tokenizer.bos_token,
            self.tokenizer.eos_token,
            self.tokenizer.pad_token,
            self.tokenizer.unk_token,
        }
        # Replace invalid characters with 'X'
        sequence = "".join([char if char in valid_chars else "X" for char in sequence])
        input_ids = self.tokenizer.encode(sequence)
        if self.block_size is not None:
            # Insert a <bos> at start, <eos> near the end, and pad if needed
            input_ids = (
                [self.tokenizer.token_dict[self.tokenizer.bos_token]] +
                input_ids[1:-1][:self.block_size - 2] +
                [self.tokenizer.token_dict[self.tokenizer.eos_token]]
            )
            padding_needed = max(0, self.block_size - len(input_ids))
            input_ids += [self.tokenizer.token_dict[self.tokenizer.pad_token]] * padding_needed

        # Convert to torch tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # The usual next-token prediction: inputs are everything but last,
        # targets are everything but first
        inputs = input_ids[:-1]
        targets = input_ids[1:]
        return inputs, targets


# -----------------------------------------------------------------------------
# Collate Function for DataLoader
# -----------------------------------------------------------------------------
def collate_fn(batch, tokenizer, block_size):
    """
    Takes a list of (inputs, targets) pairs, and pads them to uniform length
    for batch training. Also sets padding tokens in the targets to -100 so
    they don't contribute to the loss.
    """
    inputs_list, targets_list = zip(*batch)

    # Pad the inputs and targets
    padded_inputs = tokenizer.pad_sequences(inputs_list, padding_value=0, block_size=block_size)
    padded_targets = tokenizer.pad_sequences(targets_list, padding_value=0, block_size=block_size)

    # Convert to tensors
    inputs = torch.tensor(padded_inputs, dtype=torch.long)
    targets = torch.tensor(padded_targets, dtype=torch.long)

    # In PyTorch language modeling, we typically ignore the padding in the loss calculation,
    # by setting those positions to -100.
    targets[targets == 0] = -100
    return inputs, targets


# -----------------------------------------------------------------------------
# Layer Normalization
# -----------------------------------------------------------------------------
class LayerNorm(nn.Module):
    """
    Custom layer norm: simpler version without bias if not needed.
    """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


# -----------------------------------------------------------------------------
# Self Attention
# -----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """
    A standard multi-head masked self-attention mechanism with optional LoRA dimension.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, \
            "Embedding dimension must be divisible by number of heads."

        # Optional LoRA: if lora_dim > 0, it modifies the attention projection
        if config.lora_dim == 0:
            self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3, bias=False)
        else:
            self.c_attn_a = nn.Linear(config.n_embd, config.lora_dim, bias=False)
            self.c_attn_norm = LayerNorm(config.lora_dim, bias=False)
            self.c_attn_b = nn.Linear(config.lora_dim, config.n_embd * 3, bias=False)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.lora_dim = config.lora_dim
        self.is_causal = config.is_causal

    def forward(self, x):
        """
        x: (batch_size, sequence_length, embedding_size)
        We compute Q,K,V and then perform scaled dot product attention.
        """
        B, T, C = x.size()

        # If no LoRA, normal projection. Else, apply LoRA transformations
        if self.lora_dim == 0:
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        else:
            q, k, v = self.c_attn_b(self.c_attn_norm(self.c_attn_a(x))).split(self.n_embd, dim=2)

        # Reshape Q,K,V to (batch_size, heads, sequence_length, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # PyTorch >= 2.0 provides scaled_dot_product_attention
        # which can apply the causal mask for us.
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=self.is_causal
        )
        # Bring y back to (batch_size, sequence_length, embedding_size)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # A final linear projection + dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


# -----------------------------------------------------------------------------
# MLP
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    """
    A standard MLP block used after self-attention: 
    linear -> GELU -> linear -> dropout
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 2, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.n_embd * 2, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# -----------------------------------------------------------------------------
# Transformer Block
# -----------------------------------------------------------------------------
class Block(nn.Module):
    """
    A single Transformer block: LN -> Self-Attn -> LN -> MLP
    (with residual connections around each part).
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, bias=False)
        self.sa = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, bias=False)
        self.mlp = MLP(config)

    def forward(self, x):
        # First residual block: self-attention
        x = x + self.sa(self.ln1(x))
        # Second residual block: MLP
        x = x + self.mlp(self.ln2(x))
        return x


# -----------------------------------------------------------------------------
# GPT Configuration
# -----------------------------------------------------------------------------
@dataclass
class GPT_Config:
    """
    Holds key hyperparameters for building the GPT model. 
    These will be passed to the GPT constructor.
    """
    n_embd: int = 1024      # Embedding dimension
    lora_dim: int = 256     # Optional LoRA dimension for parameter-efficient training
    max_seq_len: int = 1024 # Max sequence length (context window)
    n_head: int = 32        # Number of attention heads
    n_layer: int = 32       # Number of Transformer blocks
    dropout: float = 0.1    # Dropout rate
    vocab_size: int = 32    # Size of the vocabulary (number of tokens)
    ignore_index: int = -100 # Index to ignore in loss (e.g. for padding)
    block_size: int = max_seq_len # Typically the same as max_seq_len
    seq_padding_idx: int = 0
    is_causal: bool = True  # Whether attention is causal (mask future tokens)


# -----------------------------------------------------------------------------
# GPT Model
# -----------------------------------------------------------------------------
class GPT(nn.Module):
    """
    A GPT-style language model for protein sequences, with positional embeddings,
    multiple Transformer blocks, and a final projection to predict next token.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = ProteinTokenizer(token_dict)
        # Positional embeddings map positions 0..(max_seq_len-1) to embedding vectors
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.n_embd)
        # seq_embedding maps token IDs in [0..vocab_size-1] to embedding vectors
        self.seq_embedding = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=0)
        # Final LayerNorm after stacking Transformer blocks
        self.ln = LayerNorm(config.n_embd, bias=False)

        # Create a stack of Transformer blocks
        self.transformer = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # Final linear layer to map hidden states to vocab logits
        self.project = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights for the entire model
        self.apply(self._init_weights)

        # A special initialization for c_proj weights
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # Print the number of parameters if rank=0
        print0("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        """Calculate total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        """Default initialization: normal(0, 0.02) for linear weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, seq, targets=None):
        """
        Forward pass: 
        seq is (batch_size, sequence_length) of token IDs.
        If targets is provided, we compute cross-entropy loss.
        If not, we only return logits (useful during inference/generation).
        """
        device = seq.device
        B, T = seq.size()

        # Build position IDs and gather their embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos = self.pos_embedding(pos)

        # Look up token embeddings
        seq = self.seq_embedding(seq)

        # Sum positional + token embeddings
        x = seq + pos

        # Pass through each Transformer block in turn
        for layer in self.transformer:
            x = layer(x)

        # Final layer norm
        x = self.ln(x)

        # If we have targets, compute the language modeling loss
        if exists(targets):
            logits = self.project(x)  # (B, T, vocab_size)
            # Flatten the batch and sequence dims for cross_entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.config.ignore_index
            )
        else:
            # If no targets, return the logits for the last token only
            # (useful for token-by-token generation)
            logits = self.project(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Create an AdamW optimizer, optionally fused if supported by PyTorch and device is CUDA.
        Groups parameters into decaying and non-decaying sets based on dimension.
        """
        # Filter out any parameters that aren't trainable
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Some systems support fused AdamW for speed
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = (fused_available and device_type == 'cuda')
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print0(f"using fused AdamW: {use_fused}")
        return optimizer

    @torch.inference_mode()
    def generate(self, prefix, max_size, temperature=1.0, top_k=7, rep_penalty=5, ngram_block=4):
        """
        Generates protein sequences, starting from a prefix, up to max_size tokens.
          - temperature controls randomness
          - top_k controls sampling from top k candidates
          - rep_penalty blocks repeated tokens
          - ngram_block blocks repeated n-grams
        """
        generated = prefix.clone()  # shape [B, prefix_len]
        tokens_to_generate = max_size - prefix.size(1)
        if tokens_to_generate <= 0:
            raise ValueError(
                f"Desired size {max_size} <= prefix length {prefix.size(1)}."
            )

        for _ in range(tokens_to_generate):
            # Forward pass the entire current sequence to get logits
            logits, _ = self.forward(generated, targets=None)

            # Extract logits for the last token in the sequence
            next_token_logits = logits[:, -1, :]  # shape [B, vocab_size]

            # 1) repetition penalty
            next_token_logits = adjust_logits_for_repetition(
                next_token_logits, generated, rep_penalty=rep_penalty
            )

            # 2) n-gram blocking
            next_token_logits = adjust_logits_for_ngram_blocking(
                next_token_logits, generated, n=ngram_block
            )

            # Sample the next token from the adjusted logits
            next_token = self._sample_next_token(next_token_logits, temperature, top_k)

            # Append the sampled token
            generated = torch.cat([generated, next_token], dim=1)

            # If we hit <eos>, stop generating
            if next_token.item() == self.tokenizer.eos_id:
                break

        return generated

    def _sample_next_token(self, logits, temperature=1.0, top_k=7):
        """
        Takes the logits for a single step and chooses a token from the distribution.
        - If top_k > 0, we only consider the top k tokens by logit.
        """
        # Scale by temperature
        logits = logits / temperature

        if top_k > 0:
            # Get top_k probabilities
            top_logits, top_indices = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(top_logits, dim=-1)
            # Sample from those top_k
            indices = torch.multinomial(probs, num_samples=1)
            next_tokens = top_indices.gather(-1, indices)
        else:
            # Full distribution
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

        return next_tokens


# -----------------------------------------------------------------------------
# Generation Helpers (block repeated tokens, repeated n-grams, etc.)
# -----------------------------------------------------------------------------
def adjust_logits_for_repetition(logits, generated_seq, rep_penalty=4):
    """
    If the last token in generated_seq is repeated rep_penalty times consecutively,
    we set its logit to -∞ to block it.
    """
    logits = logits.clone()
    B = logits.size(0)

    for i in range(B):
        seq = generated_seq[i]
        if seq.numel() == 0:
            continue
        last_token = seq[-1].item()
        count = 1
        j = seq.size(0) - 2
        # Count how many times the last token repeats at the end
        while j >= 0 and seq[j].item() == last_token:
            count += 1
            j -= 1
        if count >= rep_penalty:
            logits[i, last_token] = float('-inf')
    return logits

def adjust_logits_for_ngram_blocking(logits, generated_seq, n=3):
    """
    If appending a candidate token would form an n-gram that already appeared in generated_seq,
    we set its logit to -∞ to block it.
    """
    logits = logits.clone()
    B = logits.size(0)
    for i in range(B):
        seq = generated_seq[i]
        if seq.size(0) < n - 1:
            continue

        # Get the last (n - 1) tokens
        context = tuple(seq[-(n - 1):].tolist())
        banned_tokens = set()

        # Scan through the sequence to find repeating n-grams
        for start_idx in range(seq.size(0) - (n - 1)):
            window = seq[start_idx:start_idx + (n - 1)]
            if tuple(window.tolist()) == context:
                # The token that followed that n-1 context is the banned token
                if start_idx + (n - 1) < seq.size(0):
                    banned_tokens.add(seq[start_idx + (n - 1)].item())

        # Block all banned tokens
        for token in banned_tokens:
            logits[i, token] = float('-inf')
    return logits


# -----------------------------------------------------------------------------
# Main Training Script
# -----------------------------------------------------------------------------
tokenizer = ProteinTokenizer(token_dict)
parser = argparse.ArgumentParser(description="Train a LLaMA model on protein sequences.")

# Dataset arguments
parser.add_argument(
    '--fasta_file', type=str, default="/scratch/mnaseri1/seq/uniprot_trembl_t.fasta",
    help='Path to the FASTA file containing protein sequences.'
)
parser.add_argument(
    '--block_size', type=int, default=GPT_Config.block_size,
    help='Maximum sequence length (block size) to train on.'
)
parser.add_argument(
    '--val_fraction', type=float, default=0.0001,
    help='Fraction of total data used for validation.'
)

# Training arguments
parser.add_argument(
    '--init_from', type=str, choices=['scratch', 'resume'], default='resume',
    help='Initialize model from scratch or resume from a saved checkpoint.'
)
parser.add_argument('--eval_only', default='False', help='If set, only run evaluation.')
parser.add_argument('--grad_clip', type=float, default=1.0, help='Clip gradients above this norm.')
parser.add_argument('--backend', type=str, default='nccl', help='Backend for DDP (nccl/gloo/etc).')
parser.add_argument(
    '--out_dir', type=str, default='out_pretrain_GPT_big',
    help='Directory to save model checkpoints and logs.'
)
parser.add_argument(
    '--dtype', type=str, choices=['float32', 'bfloat16', 'float16'],
    default='bfloat16',
    help='Data precision for model weights and activations.'
)
parser.add_argument('--amp', default=True, help='Use Automatic Mixed Precision if True.')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size per GPU.')
parser.add_argument('--grad_accum_steps', type=int, default=2, help='Accumulate gradients this many steps before backward.')
parser.add_argument('--weight_decay', type=float, default=1e-1, help='Weight decay factor.')
parser.add_argument('--lr', type=float, default=6e-4, help='Initial learning rate.')
parser.add_argument('--min_lr', type=float, default=5e-6, help='Minimum learning rate after decay.')
parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1.')
parser.add_argument('--beta2', type=float, default=0.95, help='Adam beta2.')
parser.add_argument('--warmup_iters', type=int, default=2000, help='Steps of warmup for LR.')
parser.add_argument(
    '--lr_decay_iters', type=int, default=200000,
    help='Total steps of learning rate cosine decay.'
)
parser.add_argument(
    '--eval_interval', type=int, default=1000,
    help='Evaluate the model on the validation set every this many steps.'
)
parser.add_argument('--eval_iters', type=int, default=100, help='Number of mini-batches to eval.')
parser.add_argument('--always_save_checkpoint', default=True, help='Save checkpoint after each eval.')
parser.add_argument('--log_interval', type=int, default=1, help='Log training loss every n steps.')
parser.add_argument('--max_iter', type=int, default=300000, help='Total steps to train.')

# Distributed training arguments
parser.add_argument('--ddp', default=True, help='Use Distributed Data Parallel if True.')

args = parser.parse_args()

# -----------------------------------------------------------------------------
# Distributed Setup
# -----------------------------------------------------------------------------
if args.ddp:
    init_process_group(backend=args.backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master = (ddp_rank == 0)
    seed_offset = ddp_rank
else:
    master = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_rank = 0
    ddp_local_rank = 0

# Create output directory if on main process
if master:
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"results will be saved at {args.out_dir}!")

# Set random seeds
torch.manual_seed(2001 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Choose device and precision
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16
}[args.dtype]
ctx = nullcontext() if device_type == "cpu" else torch.autocast(
    enabled=args.amp, dtype=torch.bfloat16, device_type="cuda"
)

# -----------------------------------------------------------------------------
# Create Datasets and DataLoaders
# -----------------------------------------------------------------------------
train_dataset = ProteinIterableDataset(
    fasta_file=args.fasta_file,
    tokenizer=tokenizer,
    block_size=args.block_size,
    shuffle_buffer_size=2**10,
    split='train',
    val_fraction=args.val_fraction,
    rank=ddp_rank,
    world_size=ddp_world_size
)
val_dataset = ProteinIterableDataset(
    fasta_file=args.fasta_file,
    tokenizer=tokenizer,
    block_size=args.block_size,
    shuffle_buffer_size=2**10,
    split='val',
    val_fraction=args.val_fraction,
    rank=ddp_rank,
    world_size=ddp_world_size
)

# We use a custom collate_fn that pads sequences
collate_fn_with_tokenizer = partial(collate_fn, tokenizer=tokenizer, block_size=args.block_size)

# DataLoader for training
dl = DataLoader(
    train_dataset, batch_size=args.batch_size,
    collate_fn=collate_fn_with_tokenizer,
    num_workers=0, pin_memory=True, drop_last=True
)
# DataLoader for validation
val_dl = DataLoader(
    val_dataset, batch_size=args.batch_size,
    collate_fn=collate_fn_with_tokenizer,
    num_workers=0, pin_memory=True, drop_last=True
)

# -----------------------------------------------------------------------------
# Initialize/Resume the Model
# -----------------------------------------------------------------------------
iter_num = 0
best_val_loss = 1e9

model_args = dict(
    n_layer=GPT_Config.n_layer,
    n_head=GPT_Config.n_head,
    n_embd=GPT_Config.n_embd,
    vocab_size=GPT_Config.vocab_size,
    dropout=GPT_Config.dropout,
    seq_padding_idx=GPT_Config.seq_padding_idx,
    block_size=GPT_Config.block_size,
    max_seq_len=GPT_Config.max_seq_len,
    lora_dim=GPT_Config.lora_dim,
    ignore_index=GPT_Config.ignore_index,
    is_causal=GPT_Config.is_causal
)
print0(model_args)

if args.init_from == "scratch":
    # Create a new model from scratch
    print("Training new model from scratch!")
    config = GPT_Config(**model_args)
    model = GPT(config)

if args.init_from == "resume":
    # Load from existing checkpoint
    print(f"Resuming training from {args.out_dir}")
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)

    cfg = GPT_Config(**model_args)
    model = GPT(cfg)

    state_dict = checkpoint["model"]
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys after loading state dict: {missing_keys}")
        print("Likely new layers, randomly initialized.")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]

# Move model to GPU/CPU
print0("World size:", ddp_world_size)
model.to(device)

# Create optimizer (configured inside the model) and maybe load its state
optimizer = model.configure_optimizers(
    args.weight_decay, args.lr, (args.beta1, args.beta2), device_type=device_type
)
if args.init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None

# Automatic Mixed Precision scaler
scaler = torch.GradScaler(enabled=args.amp)

# Wrap model in DDP if requested
if args.ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)


# -----------------------------------------------------------------------------
# Evaluation Function
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    """
    Run a few mini-batches of validation to get an average loss,
    for both train and val sets. This helps monitor for overfitting.
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = []
        data_loader = dl if split == 'train' else val_dl
        for idx, (X, Y) in enumerate(data_loader):
            if idx >= args.eval_iters:
                break
            X = X.to(device)
            Y = Y.to(device)
            with ctx:
                _, loss = model(seq=X, targets=Y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses) if len(losses) > 0 else float('inf')
    model.train()
    return out


def get_lr(it, warmup_iters=2000, lr=6e-4, min_lr=5e-6):
    """
    Get the learning rate at iteration `it` using a simple warmup + cosine decay schedule.
    """
    if it < warmup_iters:
        # Linear warmup from 0 to lr
        return lr * (it + 1) / (warmup_iters + 1)
    if it > args.lr_decay_iters:
        # After lr_decay_iters, stay at min_lr
        return min_lr
    # Cosine decay from lr to min_lr
    decay_ratio = (it - warmup_iters) / (args.lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


# Estimate how many tokens we see in one iteration
tokens_per_iter = args.grad_accum_steps * ddp_world_size * args.batch_size * args.block_size
print0(f"tokens per iteration will be: {tokens_per_iter:,}")

# -----------------------------------------------------------------------------
# Quick Test Dataloader
# -----------------------------------------------------------------------------
# Just to confirm the DataLoader yields valid data
t0 = time.time()
raw_model = model.module if args.ddp else model

seq, targets = next(iter(dl))
seq = seq.to(device)
targets = targets.to(device)

# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------
while True:
    # We reset gradient accumulators
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)

    # Go through one epoch's worth of data. In an IterableDataset, we keep streaming.
    for X, Y in dl:
        # Move batch to device
        X = X.to(device)
        Y = Y.to(device)

        # Forward pass under AMP
        with ctx:
            _, loss = model(seq=X, targets=Y)
            # Scale loss by grad_accum_steps so we effectively 'accumulate gradients'
            loss = loss / args.grad_accum_steps

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        micro_step += 1

        if micro_step % args.grad_accum_steps == 0:
            # Update learning rate
            lr = get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # If DDP is used, sync gradients across processes
            if args.ddp:
                model.require_backward_grad_sync = True

            # Optionally clip gradients to avoid exploding grads
            if args.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # Take an optimizer step
            scaler.step(optimizer)
            scaler.update()

            # Clear gradients for next accumulation
            optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            # Logging
            if iter_num % args.log_interval == 0 and master:
                lossf = loss.item() * args.grad_accum_steps
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms")

            iter_num += 1

            # Periodically evaluate and possibly save checkpoint
            if (iter_num % args.eval_interval == 0 or iter_num == 1) and master:
                losses = estimate_loss()
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                # If improved, or if user wants to always save
                if losses['val'] < best_val_loss or args.always_save_checkpoint:
                    best_val_loss = losses["val"]
                    print(f"saving checkpoint to {args.out_dir}")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': model_args,
                        'seq_idx': train_dataset.seq_idx
                    }
                    torch.save(checkpoint, os.path.join(args.out_dir, 'ckpt.pt'))

                # Demonstration of generating a few sample sequences
                print("Generating samples...")
                sample_prefixes = ["MTLP", "MTL", "MDE", "MH", "KLL", "MRIE"]
                sizes = [550, 905, 393, 300, 300, 480]
                sample_prefixes_unc = ["M", "M", "M"]
                sizes_unc = [250, 200, 400]
                with torch.no_grad():
                    with ctx:
                        # Generate conditionally, starting from specific prefixes
                        for prefix, size in zip(sample_prefixes, sizes):
                            prefix_ids = tokenizer.encode(prefix)
                            prefix_tensor = torch.tensor(
                                prefix_ids, dtype=torch.long, device=device
                            )[None, ...]
                            cond_input = (
                                prefix_tensor
                                if prefix_tensor.size(1) <= args.block_size
                                else prefix_tensor[:, -args.block_size:]
                            )
                            generated_cond = raw_model.generate(
                                prefix=cond_input,
                                max_size=size,
                                temperature=0.9,
                                top_k=5
                            )
                            generated_cond_text = tokenizer.decode(
                                generated_cond[0].cpu().tolist()
                            )
                            print(
                                f"Conditional Sample starting with {prefix}, "
                                f"generated_size = {len(generated_cond_text)}:\n"
                                f"{generated_cond_text}\n---------------"
                            )

                        # Generate unconditionally (or with a minimal prefix)
                        for prefix, size in zip(sample_prefixes_unc, sizes_unc):
                            prefix_ids = tokenizer.encode(prefix)
                            prefix_tensor = torch.tensor(
                                prefix_ids, dtype=torch.long, device=device
                            )[None, ...]
                            unc_input = (
                                prefix_tensor
                                if prefix_tensor.size(1) <= args.block_size
                                else prefix_tensor[:, -args.block_size:]
                            )
                            generated_unc = raw_model.generate(
                                prefix=unc_input,
                                max_size=size,
                                temperature=0.9,
                                top_k=5
                            )
                            generated_unc_text = tokenizer.decode(
                                generated_unc[0].cpu().tolist()
                            )
                            print(
                                f"Unconditional Sample starting with {prefix}, "
                                f"generated_size = {len(generated_unc_text)}:\n"
                                f"{generated_unc_text}\n---------------"
                            )



    # If we've reached the max iterations, stop
    if iter_num >= args.max_iter:
        break

# Clean up distributed training if needed
if args.ddp:
    destroy_process_group()

# -----------------------------------------------------------------------
# Training Complete
# -----------------------------------------------------------------------
