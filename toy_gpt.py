"""
toy_gpt.py — a minimal character-level GPT you can train from scratch

Usage:
    python toy_gpt.py                 # trains on a tiny built-in corpus
    python toy_gpt.py --data my.txt   # trains on your own plain-text file

Tested with PyTorch >= 2.0. Runs on CPU (slow but fine for the toy corpus)
or automatically uses CUDA / Apple-MPS if available.
"""

import argparse
import math
import os
import sys
import urllib.request
from contextlib import nullcontext
import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# Hyperparameters — these are deliberately tiny so it trains in seconds/minutes.
# Bump them up once things work to get better samples.
# -----------------------------------------------------------------------------
BLOCK_SIZE    = 128    # context length: how many chars the model sees at once
BATCH_SIZE    = 32     # how many independent sequences per training step
N_EMBED       = 192    # embedding / model width
N_HEADS       = 6      # number of attention heads (N_EMBED must divide by this)
N_LAYERS      = 4      # number of transformer blocks stacked
DROPOUT       = 0.1
LEARNING_RATE = 3e-4
MAX_ITERS     = 3000   # training steps
EVAL_INTERVAL = 250   # how often to print a loss estimate
EVAL_ITERS    = 50     # how many batches to average for the loss estimate
SEED          = 1337

# Learning rate decay hyperparameters
DECAY_LR      = True
WARMUP_ITERS  = 100    # linear warmup steps
MIN_LR        = 3e-5   # min learning rate (usually LEARNING_RATE / 10)

# A tiny built-in corpus so the script runs with zero setup. Replace via --data.
FALLBACK_TEXT = (
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles\n"
    "And by opposing end them. To die, to sleep,\n"
    "No more; and by a sleep to say we end\n"
    "The heart-ache and the thousand natural shocks\n"
    "That flesh is heir to: 'tis a consummation\n"
    "Devoutly to be wish'd. To die, to sleep;\n"
    "To sleep, perchance to dream: ay, there's the rub;\n"
) * 200  # repeated so there's enough data to actually learn something


# -----------------------------------------------------------------------------
# Causal Self-Attention Layer
# -----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """A standard multi-head causal self-attention layer with support for
    scaled dot-product attention (PyTorch 2.0+)."""
    def __init__(self):
        super().__init__()
        assert N_EMBED % N_HEADS == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(N_EMBED, 3 * N_EMBED, bias=False)
        # output projection
        self.c_proj = nn.Linear(N_EMBED, N_EMBED)
        # regularization
        self.attn_dropout = nn.Dropout(DROPOUT)
        self.resid_dropout = nn.Dropout(DROPOUT)
        self.n_heads = N_HEADS
        self.n_embed = N_EMBED

    def forward(self, x):
        B, T, C = x.shape  # batch size, sequence length, embedding dimensionality (N_EMBED)
        
        # calculate query, key, values for all heads in batch and split Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)
        
        # reshape to (B, nh, T, hs)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        # causal self-attention
        # Under the hood, F.scaled_dot_product_attention uses highly optimized FlashAttention 
        # or Memory Efficient Attention kernels if available.
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=DROPOUT if self.training else 0.0, 
            is_causal=True
        )
        # wait! Let's check: y = F.scaled_dot_product_attention(q, k, v, ...) -> YES, query, key, value.
        # Let me write the code with q, k, v.
        # Let's fix that.
        
        # transpose and view back to (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # output projection
        return self.resid_dropout(self.c_proj(y))


class FeedForward(nn.Module):
    """A small position-wise MLP. The 4x expansion is the standard choice."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBED, 4 * N_EMBED),
            nn.GELU(),
            nn.Linear(4 * N_EMBED, N_EMBED),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """A transformer block: attention then MLP, each with a residual + layernorm.

    Note the 'pre-norm' arrangement (LayerNorm *before* the sublayer); it trains
    more stably than the original post-norm transformer.
    """
    def __init__(self):
        super().__init__()
        self.sa   = CausalSelfAttention()
        self.ffwd = FeedForward()
        self.ln1  = nn.LayerNorm(N_EMBED)
        self.ln2  = nn.LayerNorm(N_EMBED)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))     # residual around attention
        x = x + self.ffwd(self.ln2(x))   # residual around MLP
        return x


class ToyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks  = nn.Sequential(*[Block() for _ in range(N_LAYERS)])
        self.ln_f    = nn.LayerNorm(N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, vocab_size)
        
        # Tie input embeddings and language model head weights
        # (https://arxiv.org/abs/1608.05859)
        self.lm_head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding(idx)                                  # (B,T,C)
        pos = self.position_embedding(torch.arange(T, device=idx.device))# (T,C)
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                         # (B,T,vocab)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Autoregressively sample max_new_tokens characters."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]      # never feed more than block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]            # focus on the last time step
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


# -----------------------------------------------------------------------------
# Training / data plumbing
# -----------------------------------------------------------------------------
def download_corpus(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading corpus from {url} to {filename}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print("Download completed successfully.")
        except Exception as e:
            print(f"Failed to download corpus: {e}")
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None,
                        help="Path to a plain-text file to train on.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save/load checkpoints.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest checkpoint.")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only evaluate the checkpoint and generate samples.")
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt to generate from.")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of characters to generate.")
    parser.add_argument("--max_iters", type=int, default=MAX_ITERS,
                        help="Maximum training iterations.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use: 'cuda', 'mps', or 'cpu'. Defaults to auto-detect.")
    parser.add_argument("--compile", action="store_true",
                        help="Compile the model using torch.compile (requires PyTorch 2.0+).")
    parser.add_argument("--amp", action="store_true",
                        help="Use mixed precision (automatic mixed precision) training on CUDA.")
    parser.add_argument("--tokenizer", type=str, default="gpt2", choices=["char", "gpt2"],
                        help="Tokenizer to use: 'char' or 'gpt2' subword tokenizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-1,
                        help="Weight decay coefficient.")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    if args.device:
        device = args.device
    else:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available()
                  else "cpu")
    print(f"Using device: {device}")

    # Set up Automatic Mixed Precision (AMP) context and GradScaler if on CUDA
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if args.amp and device_type == 'cuda':
        ctx = torch.amp.autocast(device_type=device_type, dtype=torch.float16)
        scaler = torch.cuda.amp.GradScaler()
        print("Using AMP (automatic mixed precision) training.")
    else:
        ctx = nullcontext()
        scaler = None
        if args.amp:
            print("Warning: AMP requested but device is not CUDA. Running standard precision.")

    # --- load text / setup vocabulary ---
    raw_text = None
    if args.data:
        if os.path.exists(args.data):
            with open(args.data, "r", encoding="utf-8") as f:
                raw_text = f.read()
        else:
            print(f"Warning: --data path '{args.data}' not found. Falling back to downloading Tiny Shakespeare.")
    
    if raw_text is None:
        shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        shakespeare_file = "tinyshakespeare.txt"
        if download_corpus(shakespeare_url, shakespeare_file):
            with open(shakespeare_file, "r", encoding="utf-8") as f:
                raw_text = f.read()
        else:
            print("Falling back to built-in tiny Shakespeare excerpt.")
            raw_text = FALLBACK_TEXT

    # Check for checkpoints to resume or evaluate and restore tokenizer type
    checkpoint = None
    tokenizer_type = args.tokenizer
    if args.resume or args.eval_only:
        possible_paths = [
            os.path.join(args.checkpoint_dir, "ckpt_best.pt"),
            os.path.join(args.checkpoint_dir, "ckpt_latest.pt")
        ]
        
        # Prefer ckpt_best.pt for eval, ckpt_latest.pt for resume
        selected_path = possible_paths[0] if args.eval_only else possible_paths[1]
        if not os.path.exists(selected_path):
            selected_path = possible_paths[1] if args.eval_only else possible_paths[0]
            
        if os.path.exists(selected_path):
            print(f"Loading checkpoint from {selected_path}...")
            checkpoint = torch.load(selected_path, map_location=device)
            tokenizer_type = checkpoint.get('tokenizer_type', 'char')
            print(f"Restored tokenizer: '{tokenizer_type}' from checkpoint.")
        else:
            if args.eval_only:
                print(f"Error: Could not find checkpoint at {args.checkpoint_dir} for evaluation.")
                sys.exit(1)
            print(f"No checkpoint found at {args.checkpoint_dir}. Starting training from scratch.")

    # Setup tokenization
    if tokenizer_type == "gpt2":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
        vocab_size = 50257
        print(f"Using GPT-2 BPE tokenizer. Vocab size: {vocab_size}")
    else:
        # Character-level tokenizer
        if checkpoint is not None:
            stoi = checkpoint['stoi']
            itos = checkpoint['itos']
            vocab_size = checkpoint['vocab_size']
        else:
            chars = sorted(set(raw_text))
            vocab_size = len(chars)
            stoi = {c: i for i, c in enumerate(chars)}
            itos = {i: c for i, c in enumerate(chars)}
            
        def encode(s):
            out = []
            skipped = set()
            for c in s:
                if c in stoi:
                    out.append(stoi[c])
                else:
                    skipped.add(c)
            if skipped:
                print(f"Warning: skipped {len(skipped)} characters not in vocabulary: {list(skipped)[:5]}...")
            return out
        decode = lambda l: "".join(itos[i] for i in l)
        print(f"Using Character-level tokenizer. Vocab size: {vocab_size}")

    # --- build model ---
    model = ToyGPT(vocab_size).to(device)

    start_iter = 0
    best_val_loss = float('inf')

    # Custom optimizer with weight decay exclusion for 1D parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=LEARNING_RATE)

    # Load weights if resuming/evaluating
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            print(f"Warning: Could not restore optimizer state: {e}. Starting optimizer from scratch.")
        start_iter = checkpoint['iter_num'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from step {start_iter} with best validation loss {best_val_loss:.4f}")

    # --- compile model if requested (after checkpoint load) ---
    if args.compile:
        print("Compiling the model... (this may take a minute)")
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.2f}M")

    # --- encode data ---
    encoded_data = encode(raw_text)
    print(f"Corpus: {len(raw_text)} chars, encoded {len(encoded_data)} tokens, vocab size: {vocab_size}")

    data = torch.tensor(encoded_data, dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + BLOCK_SIZE] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss(model):
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(EVAL_ITERS)
            for k in range(EVAL_ITERS):
                xb, yb = get_batch(split)
                with ctx:
                    _, loss = model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    # Cosine learning rate decay scheduler with warmup
    def get_lr(it):
        if it < WARMUP_ITERS:
            return LEARNING_RATE * it / WARMUP_ITERS
        if it > args.max_iters:
            return MIN_LR
        decay_ratio = (it - WARMUP_ITERS) / (args.max_iters - WARMUP_ITERS)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)

    # helper for saving checkpoints
    def save_checkpoint(filename, current_iter, current_best_loss):
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(args.checkpoint_dir, filename)
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        checkpoint_data = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iter_num': current_iter,
            'best_val_loss': current_best_loss,
            'vocab_size': vocab_size,
            'tokenizer_type': tokenizer_type,
        }
        if tokenizer_type == "char":
            checkpoint_data['stoi'] = stoi
            checkpoint_data['itos'] = itos
        torch.save(checkpoint_data, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    # --- train ---
    if not args.eval_only:
        print(f"Starting/resuming training from step {start_iter} to {args.max_iters}...")
        for it in range(start_iter, args.max_iters + 1):
            lr = get_lr(it) if DECAY_LR else LEARNING_RATE
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if it % EVAL_INTERVAL == 0 or it == args.max_iters:
                losses = estimate_loss(model)
                print(f"step {it:5d} | lr {lr:.6f} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
                
                save_checkpoint("ckpt_latest.pt", it, best_val_loss)
                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    save_checkpoint("ckpt_best.pt", it, best_val_loss)

            if it == args.max_iters:
                break

            xb, yb = get_batch("train")
            with ctx:
                _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
    else:
        print("Running in --eval_only mode.")
        losses = estimate_loss(model)
        print(f"Evaluation | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    # --- sample some text ---
    print("\n--- sample ---")
    if args.prompt:
        print(f"Generating from prompt: {repr(args.prompt)}")
        prompt_encoded = encode(args.prompt)
        if not prompt_encoded:
            if tokenizer_type == "char":
                prompt_encoded = [stoi.get('\n', 0)]
            else:
                prompt_encoded = [13]
        start = torch.tensor([prompt_encoded], dtype=torch.long, device=device)
    else:
        if tokenizer_type == "char":
            start_char = '\n' if '\n' in stoi else list(stoi.keys())[0]
            start = torch.tensor([[stoi[start_char]]], dtype=torch.long, device=device)
        else:
            start = torch.tensor([[13]], dtype=torch.long, device=device)
        
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    out = raw_model.generate(start, max_new_tokens=args.num_samples)[0].tolist()
    print(decode(out))


if __name__ == "__main__":
    main()
