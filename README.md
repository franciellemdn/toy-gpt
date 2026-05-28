# Toy GPT — Minimal GPT From Scratch (Character & Subword BPE)

This repository contains a clean, educational, and highly optimized Generative Pre-trained Transformer (GPT) model built using PyTorch. It serves as a hands-on learning tool for understanding modern transformer architectures and GPT-style pre-training.

---

## 📂 Project Structure

The project is organized into modular files to keep the model code, dataset handling, configuration, and execution flows clean and separate:

```text
toy-gpt/
├── toy_gpt/                      # Core module package
│   ├── __init__.py               # Re-exports key classes
│   ├── config.py                 # Hyperparameter defaults (embed sizes, LR settings)
│   ├── data.py                   # Corpus downloads, BPE and Character tokenizers
│   └── model.py                  # PyTorch blocks (CausalSelfAttention, Block, ToyGPT)
│
├── train.py                      # Main training script (runs CUDA, AMP, compile)
├── sample.py                     # Main generation/sampling script
├── toy_gpt.py                    # Backwards-compatible legacy training wrapper
└── requirements.txt              # Package dependencies (PyTorch, tiktoken)
```

---

## 🚀 Key Features

* **Subword Tokenization (BPE)**: Supports both character-level tokenization and subword Byte-Pair Encoding (BPE) using OpenAI's `tiktoken` (defaults to the `gpt2` vocabulary of 50,257 tokens).
* **Automatic Corpus Downloader**: Automatically downloads the **Tiny Shakespeare** corpus (1.1 MB) if no local dataset file is specified, giving you a real, rich text source out-of-the-box.
* **Architectural Upgrades**:
  - **Weight Tying**: Shares parameters between the token embedding matrix and final output linear projection layer (`lm_head`), saving memory and accelerating learning.
  - **Selective Weight Decay**: Excludes 1D parameter vectors (biases and LayerNorm scale parameters) from regularization, matching state-of-the-art LLM training pipelines.
  - **Gradient Clipping**: Prevents gradient explosions during training using a max norm ceiling of 1.0.
* **Highly Optimized Attention**: Utilizes unified QKV projection and PyTorch's native `F.scaled_dot_product_attention` (SDPA), leveraging FlashAttention under the hood.
* **Robust Checkpoints & Resuming**: Saves model weight states, optimizer gradients, learning curves, and exact tokenizer settings to seamlessly resume training.
* **Warmup & Cosine LR Decay**: Implements standard learning rate decay with a linear warmup phase.
* **Text Prompt Priming**: Seeds text generation with any custom sentence prompt.

---

## 🛠️ Requirements & Installation

1. Install requirements using your environment's pip:
   ```bash
   pip install -r requirements.txt
   ```
2. Run training (by default, it will automatically download Tiny Shakespeare and train using the `gpt2` subword tokenizer):
   ```bash
   python train.py
   ```
   *(Note: You can also run `python toy_gpt.py` which acts as a backwards-compatible alias wrapper for `train.py`.)*

---

## 📖 Usage Guide

### 1. Basic Training (BPE Subword Level)
Trains on Tiny Shakespeare using the `gpt2` subword tokenizer (vocab size 50,257):
```bash
python train.py
```

### 2. Basic Training (Character Level)
Trains on Tiny Shakespeare using a character-level tokenizer (vocab size 65):
```bash
python train.py --tokenizer char
```

### 3. High-Performance GPU Training (CUDA)
For fast training on modern NVIDIA GPUs, you can enable mixed precision (AMP) and model compilation:
```bash
# Train on CUDA GPU using Mixed Precision (AMP) and compilation (PyTorch 2.0+)
python train.py --device cuda --amp --compile
```
* **`--device cuda`**: Forces PyTorch to target the NVIDIA CUDA backend.
* **`--amp`**: Enables Automatic Mixed Precision (FP16), reducing GPU memory usage and increasing speeds.
* **`--compile`**: Runs `torch.compile` to fuse operations and speed up execution.

### 4. Train on Your Custom Corpus
You can supply any plain-text file using the `--data` flag:
```bash
python train.py --data path/to/my_data.txt
```

### 5. Change Training Durations & Weight Decay
You can customize model parameters directly via CLI flags:
```bash
python train.py --max_iters 5000 --weight_decay 0.05 --n_layers 6 --n_embed 256
```

### 6. Resume Interrupted Training
Automatically restores tokenizer settings and continues training from your last saved checkpoint (`ckpt_latest.pt`):
```bash
python train.py --resume --max_iters 6000
```

### 7. Generate Text (Inference Only)
To sample text without training, use the dedicated generation script `sample.py` (which automatically restores the correct model size and tokenizer from the checkpoint metadata):
```bash
python sample.py --prompt "To be, or not to be" --num_samples 300
```

---

## 📂 Checkpoints Structure

Training outputs checkpoints into a `checkpoints/` folder:
* **`ckpt_latest.pt`**: Updated every evaluation interval. Useful for resuming training.
* **`ckpt_best.pt`**: Updated only when validation loss improves. Best suited for inference and text generation.

Each checkpoint is a dictionary containing:
* `model`: Model weight state dictionary (`state_dict`).
* `optimizer`: Optimizer parameter state dictionary (`state_dict`).
* `iter_num`: The iteration training stopped at.
* `best_val_loss`: The lowest validation loss recorded.
* `tokenizer_type`: The tokenizer name used (`'gpt2'` or `'char'`).
* `vocab_size`: Total unique vocabulary/token count.
* `stoi` & `itos`: (Only saved when `--tokenizer char` is active) Character-to-integer mappings.

---

## 🎓 Learning: How GPT Learns From Scratch

1. **Tokenization**: 
   - *Character Level*: Maps each distinct character to a unique integer index (e.g. `'a'` -> `1`, `'b'` -> `2`).
   - *Subword Level (BPE)*: Groups common character chunks into subwords (e.g. `'the'`, `'ing'`) using a pre-trained dictionary, compressing text sequences by ~3x and allowing the model to process 3x more context inside the same sequence window.
2. **Embeddings**: Character/subword tokens and positional indices are converted to dense vector representations (`token_embedding` + `position_embedding`).
3. **Self-Attention**: The model reads a context window (`BLOCK_SIZE = 128` tokens) and projects them into Query, Key, and Value vectors. Every token looks backwards to calculate its relevance to previous tokens.
4. **FeedForward Network**: Transforms representations individually per position.
5. **Autoregressive Generation**: The model predicts the probability distribution of the *next* token, samples a token based on this distribution, appends it to the context, and repeats the process.

Made by Francielle Marques with Antigravity2.0
