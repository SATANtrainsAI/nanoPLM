# nanoPLM

A tiny Protein Language Model (PLM) trained from scratch.  
We demonstrate how to load protein sequences from a FASTA file, tokenize them, train a GPT-style model using PyTorch, and generate new protein sequences.  

---
## Project Overview

- **Tokenizer**: Converts protein sequences into numerical tokens using a predefined vocabulary (`token_dict`).
- **Dataset**: An `IterableDataset` (`ProteinIterableDataset`) that streams sequences from a (large) FASTA file, suitable for memory-efficient training.
- **Model**: A GPT-style transformer (`GPT` class) with:
  - Positional embeddings
  - Multiple transformer blocks
  - Optional LoRA dimension for parameter-efficient fine-tuning
- **Training**: 
  - Uses `DataLoader` for batches
  - Implements distributed training with PyTorch’s `DistributedDataParallel` (DDP)
  - Applies automatic mixed precision (AMP) for faster training in float16/bfloat16 modes (configurable)
- **Generation**: Demonstrates how to generate new protein sequences with specific constraints (temperature, top-k, n-gram blocking, repetition penalty).

---

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/) (with CUDA if you want GPU acceleration)
- (Optional) Additional libraries for distributed training (e.g., `nccl` backend)

---

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/nanoPLM.git
   cd nanoPLM
