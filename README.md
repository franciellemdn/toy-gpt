# Toy GPT — Minimal Character-Level GPT From Scratch

This repository contains a clean, educational, and optimized character-level Generative Pre-trained Transformer (GPT) model built using PyTorch. It is designed to run locally, train in minutes, and serve as a hands-on learning tool for understanding modern transformer architectures.

---

## 🚀 Key Features

* **Modern Transformer Architecture**: Uses pre-layer normalization (Pre-LN) for stable training, multi-head causal self-attention, and position-wise FeedForward networks.
* **Highly Optimized Attention**: Utilizes a single unified QKV projection and PyTorch's native `F.scaled_dot_product_attention` (SDPA), which automatically leverages FlashAttention or Memory-Efficient kernels.
* **Robust Checkpoints & Resuming**: Saves model states, optimizer gradients, and exact character-to-index mappings (`stoi`/`itos`) at validation intervals.
* **Warmup & Cosine LR Decay**: Implements standard learning rate decay with a linear warmup phase to match professional LLM training scripts.
* **Text Prompt Priming**: Supports starting the text generation process with any custom text prompt.

---

## 🛠️ Requirements & Installation

1. Make sure you have PyTorch 2.0 or newer installed.
2. Run the script using your Python environment:
   ```bash
   python toy_gpt.py
   ```

*(If you are running inside a Conda environment, run:)*
```bash
conda run -n base python toy_gpt.py
```

---

## 📖 Usage Guide

### 1. Basic Training
Train on the built-in Shakespeare corpus using default hyperparameters (automatically auto-detects CUDA/MPS/CPU):
```bash
python toy_gpt.py
```

### 2. High-Performance GPU Training (CUDA)
For fast training on modern NVIDIA GPUs, you can enable mixed precision (AMP) and model compilation:
```bash
# Train on CUDA GPU using Mixed Precision (AMP) and compilation (PyTorch 2.0+)
python toy_gpt.py --device cuda --amp --compile
```
* **`--device cuda`**: Forces PyTorch to target the NVIDIA CUDA backend.
* **`--amp`**: Enables Automatic Mixed Precision (FP16), which reduces GPU memory usage and increases computing speeds.
* **`--compile`**: Runs `torch.compile` to fuse operations and speed up execution (takes ~1 minute to compile on start).

### 3. Train on Your Custom Corpus
You can supply any plain-text file (e.g. a book, code dataset, or chat logs) using the `--data` flag:
```bash
python toy_gpt.py --data path/to/my_data.txt
```

### 4. Change Training Durations
Adjust training steps using the `--max_iters` parameter:
```bash
python toy_gpt.py --max_iters 5000
```

### 5. Resume Interrupted Training
If training stops, you can resume from your last saved checkpoint (`ckpt_latest.pt`) by passing the `--resume` flag:
```bash
python toy_gpt.py --resume --max_iters 6000
```

### 6. Generate Text with Custom Prompts (Inference Only)
To sample text without running training, use `--eval_only` and prime the model with a custom sentence:
```bash
python toy_gpt.py --eval_only --prompt "To be, or not to be" --num_samples 300
```

---

## 📂 Checkpoints Structure

Training outputs checkpoints into a `checkpoints/` folder:
* **`ckpt_latest.pt`**: Updated every evaluation interval. Useful for resuming training.
* **`ckpt_best.pt`**: Updated only when validation loss improves. Best suited for inference and text generation.

Each checkpoint is a dictionary containing:
* `model`: Model state dictionary (`state_dict`).
* `optimizer`: Optimizer state dictionary (`state_dict`).
* `iter_num`: The iteration training stopped at.
* `best_val_loss`: The lowest validation loss recorded.
* `stoi` & `itos`: Character-to-integer mappings to ensure character consistency when generating.
* `vocab_size`: Total unique character count.

---

## 🎓 Learning: How GPT Learns From Scratch

1. **Tokenization**: It maps every unique character to a unique integer index (`stoi`).
2. **Embeddings**: Character tokens and positional indices are converted to dense vector representations (`token_embedding` + `position_embedding`).
3. **Self-Attention**: The model reads a context window (`BLOCK_SIZE = 128` characters) and projects them into Query, Key, and Value vectors. Every token looks backwards to calculate its relevance to previous tokens.
4. **FeedForward Network**: Transforms representations individually per position.
5. **Autoregressive Generation**: The model predicts the probability distribution of the *next* character, selects a character based on this distribution, appends it to the context, and repeats the process.

Made by Francielle Marques with Antigravity2.0
