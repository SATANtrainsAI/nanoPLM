"""
Credit to https://github.com/karpathy/nanoGPT

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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

def exists(val):
    return val is not None

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


token_dict =  {
    "<pad>": 0,"<bos>": 1,"<eos>": 2,"<unk>": 3,"A": 4,"B": 5,"C": 6,"D": 7,"E": 8,"F": 9,"G": 10,"H": 11,"I": 12,"J": 13,"K": 14, "L": 15,"M": 16,"N": 17,"O": 18,"P": 19,"Q": 20, "R": 21,"S": 22,"T": 23,"U": 24, "V": 25,"W": 26,"X": 27,"Y": 28,"Z": 29, "1": 30,"2": 31
}
token_dict_inv =  {v: k for k, v in token_dict.items()}

class ProteinTokenizer:
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
        tokens = list(sequence)
        return tokens

    def convert_tokens_to_ids(self, tokens):
        ids = [
            self.token_dict.get(token, self.token_dict[self.unk_token]) for token in tokens
        ]
        return ids

    def encode(self, sequence, add_special_tokens=True):
        tokens = self.tokenize(sequence)
        if add_special_tokens:
            tokens = [self.bos_token] + tokens 
        token_ids = self.convert_tokens_to_ids(tokens)
        return token_ids

    def decode(self, token_ids):
        tokens = [self.inv_token_dict[token_id] for token_id in token_ids]
        sequence = "".join(tokens)
        return sequence

    def pad_sequences(self, sequences, padding_value=None, block_size=None):
        if block_size is None:
            block_size = max(len(seq) for seq in sequences)
        if padding_value is None:
            padding_value = self.token_dict[self.pad_token]
        padded_sequences = [
            list(seq)[:block_size] + [padding_value] * max(0, block_size - len(seq)) for seq in sequences
        ]
        return padded_sequences

# Custom IterableDataset to read sequences from a FASTA file on the fly
class ProteinIterableDataset(IterableDataset):
    def __init__(
        self,
        fasta_file,
        tokenizer,
        block_size=1024,
        shuffle_buffer_size=10000,
        split='train',
        val_fraction=0.0001,
        rank=0,  # Add rank and world_size parameters
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

        # Set total_sequences to the actual number of sequences
        self.total_sequences = 240_000_000  # Adjusted according to your dataset size
        self.val_threshold = int(self.total_sequences * self.val_fraction)
        self.seq_idx = -1

    def __iter__(self):
        buffer = []
        with open(self.fasta_file, 'r') as f:
            seq = ''
            header = ''
            self.seq_idx = -1  # Initialize sequence index
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    self.seq_idx += 1  # Increment sequence index at each new header
                    if seq:
                        if self._is_in_split(self.seq_idx):
                            if self.split == 'train' and self.seq_idx % self.world_size != self.rank:
                                # Skip sequences not assigned to this rank
                                seq = ''
                                continue
                            
                            buffer.append(seq)
                            if len(buffer) >= self.shuffle_buffer_size:
                                random.shuffle(buffer)
                                for buffered_seq in buffer:
                                    yield self.process_sequence(buffered_seq)
                                buffer = []
                        seq = ''
                    header = line
                else:
                    seq += line.strip().upper()
            if seq:
                if self._is_in_split(self.seq_idx):
                    if self.split == 'train' and self.seq_idx % self.world_size != self.rank:
                        # Skip sequences not assigned to this rank
                        seq = ''
                    else:
                        buffer.append(seq)
            if buffer:
                random.shuffle(buffer)
                for buffered_seq in buffer:
                    yield self.process_sequence(buffered_seq)
    
    def __len__(self):
        # Approximate the length as total_sequences divided by the world size
        if self.split == 'train':
            train_sequences = self.total_sequences - self.val_threshold
            return train_sequences // self.world_size
        elif self.split == 'val':
            return self.val_threshold // self.world_size
        else:
            return 0

    def _is_in_split(self, seq_idx):
        if self.split == 'train':
            return seq_idx >= self.val_threshold
        elif self.split == 'val':
            return seq_idx < self.val_threshold
        else:
            return False

    def process_sequence(self, sequence):
        valid_chars = set(self.tokenizer.token_dict.keys()) - {
            self.tokenizer.bos_token,
            self.tokenizer.eos_token,
            self.tokenizer.pad_token,
            self.tokenizer.unk_token,
        }
        sequence = "".join([char if char in valid_chars else "X" for char in sequence])
        input_ids = self.tokenizer.encode(sequence)
        if self.block_size is not None:
            input_ids = (
                [self.tokenizer.token_dict[self.tokenizer.bos_token]] +
                input_ids[1:-1][:self.block_size - 2] +
                [self.tokenizer.token_dict[self.tokenizer.eos_token]]
            )
            input_ids = input_ids + [self.tokenizer.token_dict[self.tokenizer.pad_token]] * max(0, self.block_size - len(input_ids))

        input_ids = torch.tensor(input_ids, dtype=torch.long)

        inputs = input_ids[:-1]
        targets = input_ids[1:]

        return inputs, targets

# Collate function for DataLoader to handle variable-length sequences
def collate_fn(batch, tokenizer, block_size):
    inputs_list, targets_list = zip(*batch)
    padded_inputs = tokenizer.pad_sequences(inputs_list, padding_value=0, block_size=block_size)
    padded_targets = tokenizer.pad_sequences(targets_list, padding_value=0, block_size=block_size)
    inputs = torch.tensor(padded_inputs, dtype=torch.long)
    targets = torch.tensor(padded_targets, dtype=torch.long)
    targets[targets == 0] = -100
    
    return inputs, targets


        








#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



def exists(val):
    return val is not None


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


            


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        if config.lora_dim == 0:
            self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3, bias = False)
        else:
            self.c_attn_a = nn.Linear(config.n_embd, config.lora_dim, bias = False)
            self.c_attn_norm = LayerNorm(config.lora_dim, bias = False)
            self.c_attn_b = nn.Linear(config.lora_dim, config.n_embd * 3, bias = False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.lora_dim = config.lora_dim
        self.is_causal = config.is_causal

    def forward(self, x):
        B, T, C = x.size()
        if self.lora_dim == 0:
            q, k, v = self.c_attn(x).split(self.n_embd, dim = 2)
        else:
            q, k, v = self.c_attn_b(self.c_attn_norm(self.c_attn_a(x))).split(self.n_embd, dim = 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, h, T, C / h)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask = None, dropout_p = self.dropout if self.training else 0, is_causal = self.is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y #(B, T, C)
    

class CausalCrossAttention(nn.Module):
    def __init__(self, config, config_vit):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.lora_dim = config.lora_dim
        if config.lora_rank == 0:
            self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias = False)
            self.kv_proj = nn.Linear(config_vit.n_embd, config.n_embd * 2, bias = False)
        else:
            self.q_proj_a = nn.Linear(config.n_embd, config.lora_dim, bias = False)
            self.kv_proj_a = nn.Linear(config_vit.n_embd, config.lora_dim, bias = False)
            self.ln = LayerNorm(config.lora_dim * 2, bias = False)
            self.qkv_b = nn.Linear(config.lora_dim * 2, config.n_embd * 3, bias = False)
            
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, context):
        B, T, C = x.size()
        B, T1, C1 = context.size()
        if self.lora_dim == 0:
            q = self.q_proj(x) #(B, T, C)
            kv = self.kv_proj(context) #(B, T1, C)
            k, v = torch.split(self.n_embd, dim = -1)
        else:
            q = self.q_proj_a(x)
            kv = self.kv_proj_a(context)
            qkv = torch.cat([q, kv], dim = -1)
            q, k, v = self.qkv_b(self.ln(qkv)).split(self.n_embd, dim = -1)
        
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, h, T, C / h)
        k = k.view(B, T1, self.n_head, C // self.n_head).transpose(1, 2) #(B, h, T1, C / h)
        v = v.view(B, T1, self.n_head, C // self.n_head).transpose(1, 2) #(B, h, T1, C / h)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask = None, dropout_p = self.dropout if self.training else 0, is_causal =True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y #(B, T, C)
    


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 2, bias = False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.n_embd * 2, config.n_embd, bias = False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, bias = False)
        self.sa = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, bias = False)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x
    

class CrossAttentionBlock(nn.Module):
    def __init__(self, config, config_vit):
        self.ln1 = LayerNorm(config.n_embd, bias = False)
        self.ca = CausalCrossAttention(config, config_vit)
        self.ln2 = LayerNorm(config_vit.n_embd, bias = False)
        self.mlp = MLP(config)
        self.ln3 = LayerNorm(config.n_embd, bias = False)

    def forward(self, x, context):
        x = x + self.ca(self.ln1(x), self.ln2(context))
        x = x + self.mlp(self.ln3(x))

        return x
    






@dataclass
class GPT_Config:
    n_embd: int = 1536
    lora_dim: int = 0
    max_seq_len: int = 1024
    n_head: int = 32
    n_layer: int = 32
    dropout: float = 0.1
    vocab_size: int = 32
    ignore_index: int = -100
    block_size: int = max_seq_len
    seq_padding_idx: int = 0
    is_causal: bool = True
    clip_dim: int = 512
    n_main_layers: int = 24



class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = ProteinTokenizer(token_dict)
        
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.n_embd)
        self.seq_embedding = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=0)

        self.ln = LayerNorm(config.n_embd, bias=False)
        
        # Initialize transformer layers
        self.transformer = nn.ModuleList([
            Block(config) for _ in range(config.n_layer)
        ])

        self.project = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        
        print0("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, seq, targets=None):

        device = seq.device
        B, T = seq.size()
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos = self.pos_embedding(pos)

        seq = self.seq_embedding(seq)
        
        x = seq + pos
    
        for layer in self.transformer:
            x = layer(x)
    
        x = self.ln(x)
    
        if exists(targets):
            logits = self.project(x)  # (B, T, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.config.ignore_index
            )
        else:
            logits = self.project(x[:, [-1], :])
            loss = None
    
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print0(f"using fused AdamW: {use_fused}")

        return optimizer
    

    @torch.inference_mode()     
    def generate(self, prefix, max_size, temperature=1.0, top_k=7, 
                rep_penalty=5, ngram_block=4):
        generated = prefix.clone()
        tokens_to_generate = max_size - prefix.size(1)
        if tokens_to_generate <= 0:
            raise ValueError(f"Desired size {max_size} <= prefix length {prefix.size(1)}.")
        
        for _ in range(tokens_to_generate):
            logits, _ = self.forward(generated, targets=None)
            # Take the last-step logits
            next_token_logits = logits[:, -1, :]  # shape [B, vocab_size]
            
            # 1) Apply repetition penalty
            next_token_logits = adjust_logits_for_repetition(next_token_logits, generated, 
                                                            rep_penalty=rep_penalty)
            # 2) Apply n-gram blocking
            next_token_logits = adjust_logits_for_ngram_blocking(next_token_logits, generated, 
                                                                n=ngram_block)
            
            # Now sample from the adjusted logits
            next_token = self._sample_next_token(next_token_logits, temperature, top_k)
            generated = torch.cat([generated, next_token], dim=1)
            
            # If we hit the EOS token, break
            if next_token.item() == self.tokenizer.eos_id:
                break
        
        return generated
    
    def _sample_next_token(self, logits, temperature=1.0, top_k=7):
        logits = logits / temperature
        if top_k > 0:
            top_logits, top_indices = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(top_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=1)
            next_tokens = top_indices.gather(-1, indices)
        else:
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
    
        return next_tokens
    

def adjust_logits_for_repetition(logits, generated_seq, rep_penalty=4):
    """
    For each example in the batch, if the last token is repeated consecutively
    `rep_penalty` times or more in the generated sequence, set its logit to -∞.
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
        while j >= 0 and seq[j].item() == last_token:
            count += 1
            j -= 1
        if count >= rep_penalty:
            logits[i, last_token] = float('-inf')  # block
    return logits

def adjust_logits_for_ngram_blocking(logits, generated_seq, n=3):
    """
    For each example in the batch, if appending a candidate token would form an n-gram
    that has already appeared in the generated sequence, then set its logit to -∞.
    
    For each batch element, we take the last (n-1) tokens as context. Then we search
    over the previously generated tokens for any occurrence of that (n-1)-gram. For every
    occurrence, we add the token that followed that occurrence to a banned set.
    """
    logits = logits.clone()
    B = logits.size(0)
    for i in range(B):
        seq = generated_seq[i]
        # Not enough tokens to form an n-gram
        if seq.size(0) < n - 1:
            continue
        # The last (n-1) tokens
        context = tuple(seq[-(n - 1):].tolist())
        banned_tokens = set()
        # Search the entire sequence for the same (n-1)-gram
        for start_idx in range(seq.size(0) - (n - 1)):
            window = seq[start_idx:start_idx + (n - 1)]
            if tuple(window.tolist()) == context:
                # The token that followed this (n-1)-gram is at start_idx + (n-1)
                # but be sure we don't run off the end
                if start_idx + (n - 1) < seq.size(0):
                    banned_tokens.add(seq[start_idx + (n - 1)].item())
        # Block all banned tokens
        for token in banned_tokens:
            logits[i, token] = float('-inf')
    return logits


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


tokenizer = ProteinTokenizer(token_dict)


parser = argparse.ArgumentParser(description="Train a LLaMA model on protein sequences.")
    
    # Dataset arguments
parser.add_argument('--fasta_file', type=str, default="/scratch/mnaseri1/seq/uniprot_trembl_t.fasta", help='Path to the FASTA file.')
parser.add_argument('--block_size', type=int, default=GPT_Config.block_size, help='Maximum sequence length.')
parser.add_argument('--val_fraction', type=float, default=0.0001, help='Fraction of data to use for validation.')
    
    # Training arguments
parser.add_argument('--init_from', type=str, choices=['scratch', 'resume'], default='resume', help='Initialize from scratch or resume.')
parser.add_argument('--eval_only', default='False', help='Run evaluation only.')
parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value.')
parser.add_argument('--backend', type=str, default='nccl', help='Backend for distributed training.')
parser.add_argument('--out_dir', type=str, default='out_pretrain_GPT_big', help='Directory to save outputs and checkpoints.')
parser.add_argument('--dtype', type=str, choices=['float32', 'bfloat16', 'float16'], default='bfloat16', help='Data type for model weights.')
parser.add_argument('--amp', default=True, help='Use Automatic Mixed Precision.')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
parser.add_argument('--grad_accum_steps', type=int, default=2, help='Gradient accumulation steps.')
parser.add_argument('--weight_decay', type=float, default=1e-1, help='Weight decay factor.')
parser.add_argument('--lr', type=float, default=6e-4, help='Learning rate.')
parser.add_argument('--min_lr', type=float, default=5e-6, help='Minimum learning rate.')
parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer.')
parser.add_argument('--beta2', type=float, default=0.95, help='Beta2 for Adam optimizer.')
parser.add_argument('--warmup_iters', type=int, default=2000, help='Number of warmup iterations.')
parser.add_argument('--lr_decay_iters', type=int, default=200000, help='Number of iterations for learning rate decay.')
parser.add_argument('--eval_interval', type=int, default=1000, help='Iterations between evaluations.')
parser.add_argument('--eval_iters', type=int, default=100, help='Number of evaluation iterations.')
parser.add_argument('--always_save_checkpoint', default=True, help='Always save checkpoint.')
parser.add_argument('--log_interval', type=int, default=1, help='Iterations between logging.')
parser.add_argument('--max_iter', type=int, default=300000, help='Maximum number of training iterations.')
    
    # Distributed training arguments
parser.add_argument('--ddp', default = True, help='Use Distributed Data Parallel (DDP).')

args = parser.parse_args()

if args.ddp:
    init_process_group(backend = args.backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master = ddp_rank == 0
    seed_offset = ddp_rank

else:
    master = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_rank = 0
    ddp_local_rank = 0

if master:
    os.makedirs(args.out_dir, exist_ok = True)
    print(f"results will be saved at {args.out_dir}!")


torch.manual_seed(2001 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = "cuda" if "cuda" in device else "cpu"

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
ctx = nullcontext() if device_type == "cpu" else torch.autocast(enabled = args.amp, dtype=torch.bfloat16, device_type="cuda")

train_dataset = ProteinIterableDataset(
    fasta_file=args.fasta_file,
    tokenizer=tokenizer,
    block_size=args.block_size,
    shuffle_buffer_size=2**10,
    split='train',
    val_fraction=args.val_fraction,
    rank = ddp_rank,
    world_size=ddp_world_size
)

val_dataset = ProteinIterableDataset(
    fasta_file=args.fasta_file,
    tokenizer=tokenizer,
    block_size=args.block_size,
    shuffle_buffer_size=2**10,
    split='val',
    val_fraction=args.val_fraction,
    rank = ddp_rank,
    world_size=ddp_world_size
)

collate_fn_with_tokenizer = partial(collate_fn, tokenizer=tokenizer, block_size=args.block_size)


dl = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    collate_fn=collate_fn_with_tokenizer,
    num_workers= 0,
    pin_memory = True,
    drop_last = True
)


val_dl = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    collate_fn=collate_fn_with_tokenizer,
    num_workers= 0,
    pin_memory = True, 
    drop_last = True
)


iter_num = 0
best_val_loss = 1e9

model_args = dict(
    n_layer=GPT_Config.n_layer,
    n_head=GPT_Config.n_head,
    n_embd=GPT_Config.n_embd,
    vocab_size=GPT_Config.vocab_size,
    dropout = GPT_Config.dropout,
    seq_padding_idx = GPT_Config.seq_padding_idx,
    block_size=GPT_Config.block_size,
    max_seq_len=GPT_Config.max_seq_len,
    lora_dim = GPT_Config.lora_dim,
    ignore_index = GPT_Config.ignore_index,
    is_causal = GPT_Config.is_causal,
    clip_dim = GPT_Config.clip_dim,
    n_main_layers = GPT_Config.n_main_layers

)


print0(model_args)

if args.init_from == "scratch":
    print("Trainig new model form scratch!")
    config = GPT_Config(**model_args)
    model = GPT(config)

if args.init_from == "resume":
    print(f"Resuming training from {args.out_dir}")
    ckpt_path = os.path.join(args.out_dir, "ckpt_big.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = GPT_Config(**model_args)
    model = GPT(cfg)
    
    # Load state dict with strict=False to allow missing keys (new modalities)
    state_dict = checkpoint["model"]
    
    # Initialize missing layers (new modalities) if necessary
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys after loading state dict: {missing_keys}")
        print("These are likely the new modality layers and will be randomly initialized.")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
        # These are the new layers; they are already initialized in the model's __init__
    
    # Load optimizer state dict
    optimizer = model.configure_optimizers(args.weight_decay, args.lr, (args.beta1, args.beta2), device_type = device_type)
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]

print0("World size:", ddp_world_size)
model.to(device)
scaler = torch.GradScaler(enabled = args.amp)

optimizer = model.configure_optimizers(args.weight_decay, args.lr, (args.beta1, args.beta2), device_type = device_type)
if args.init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None

if args.ddp:
    model = DDP(model, device_ids = [ddp_local_rank], find_unused_parameters=False)



@torch.no_grad()
def estimate_loss():
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
                logits, loss = model(seq = X, targets = Y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def get_lr(it, warmup_iters = 2000, lr = 6e-4, min_lr = 5e-6):
    if it < warmup_iters:
        return lr * (it + 1) / (warmup_iters + 1)
    
    if it > args.lr_decay_iters:
        return min_lr
    
    decay_ratio = (it - warmup_iters) / (args.lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


tokens_per_iter = args.grad_accum_steps * ddp_world_size * args.batch_size * args.block_size
print0(f"tokens per iteration will be: {tokens_per_iter:,}")
    
t0 = time.time()
raw_model = model.module if args.ddp else model

seq, targets = next(iter(dl))
seq = seq.to(device)
targets = targets.to(device)

iterations_per_epoch = len(dl) // args.grad_accum_steps
num_epochs = (args.max_iter + iterations_per_epoch - 1) // iterations_per_epoch


while True
    micro_step = 0

    optimizer.zero_grad(set_to_none = True)

    for X, Y in dl:
        X = X.to(device)
        Y = Y.to(device)
        

        with ctx:
            logits, loss = model(seq = X, targets = Y)
            loss = loss / args.grad_accum_steps

        scaler.scale(loss).backward()

        micro_step += 1

        if micro_step % args.grad_accum_steps == 0:
            lr = get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            if args.ddp:
                model.require_backward_grad_sync = True

            if args.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none = True)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            if iter_num % args.log_interval == 0 and master:
                lossf = loss.item() * args.grad_accum_steps
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms")

            iter_num += 1

            if iter_num % args.eval_interval == 0 or iter_num == 1 and master :
                losses = estimate_loss()
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

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
                    torch.save(checkpoint, os.path.join(args.out_dir, 'ckpt_big.pt'))
                print("Generating samples...")
                sample_prefixes = [
                    "MTLP",
                    "MTL",
                    "MDE",
                    "MH",
                    "KLL",
                    "MRIE"
                ]
                sizes = [550, 905, 393, 300, 300, 480]
                sample_prefixes_unc = ["M", "M", "M"]
                sizes_unc = [250, 200, 400]
                with torch.no_grad():
                    with ctx:
                        for prefix, size in zip(sample_prefixes, sizes):
                            prefix_ids = tokenizer.encode(prefix)
                            prefix_tensor = torch.tensor(prefix_ids, dtype = torch.long, device = device)[None, ...]
                            cond_input = prefix_tensor if prefix_tensor.size(1) <= args.block_size else prefix_tensor[:, -args.block_size:]
                            generated_cond = raw_model.generate(
                                prefix = cond_input,
                                max_size = size,
                                temperature = 0.9,
                                top_k = 5)
                            generated_cond_text = tokenizer.decode(generated_cond[0].cpu().tolist())
                            print(f"Conditional Sample starting with {prefix}, generated_size = {len(generated_cond_text)}:\n{generated_cond_text}\n---------------")
                        
                        for prefix, size in zip(sample_prefixes_unc, sizes_unc):
                            prefix_ids = tokenizer.encode(prefix)
                            prefix_tensor = torch.tensor(prefix_ids, dtype = torch.long, device = device)[None, ...]
                            unc_input = prefix_tensor if prefix_tensor.size(1) <= args.block_size else prefix_tensor[:, -args.block_size:]
                            generated_unc = raw_model.generate(
                                prefix = unc_input,
                                max_size = size,
                                temperature = 0.9,
                                top_k = 5)
                            generated_unc_text = tokenizer.decode(generated_unc[0].cpu().tolist())
                            print(f"Unconditional Sample starting with {prefix}, generated_size = {len(generated_unc_text)}:\n{generated_unc_text}\n---------------")

                        
            
            if iter_num >= args.max_iter:
                break
        
if args.ddp:
    destroy_process_group()



########################################################################
