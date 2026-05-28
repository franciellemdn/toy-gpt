import argparse
import math
import os
import sys
from contextlib import nullcontext
import torch

from toy_gpt import ToyGPT, get_tokenizer, download_corpus, config

def main():
    parser = argparse.ArgumentParser(description="Train a Toy GPT model from scratch.")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to a plain-text file to train on.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save/load checkpoints.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest checkpoint.")
    parser.add_argument("--max_iters", type=int, default=config.MAX_ITERS,
                        help="Maximum training iterations.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use: 'cuda', 'mps', or 'cpu'. Defaults to auto-detect.")
    parser.add_argument("--compile", action="store_true",
                        help="Compile the model using torch.compile (requires PyTorch 2.0+).")
    parser.add_argument("--amp", action="store_true",
                        help="Use mixed precision (automatic mixed precision) training on CUDA.")
    parser.add_argument("--tokenizer", type=str, default="gpt2", choices=["char", "gpt2"],
                        help="Tokenizer to use: 'char' or 'gpt2' subword tokenizer.")
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY,
                        help="Weight decay coefficient.")
    
    # Allow overriding model dimensions
    parser.add_argument("--n_embed", type=int, default=config.N_EMBED, help="Embedding dimension")
    parser.add_argument("--n_heads", type=int, default=config.N_HEADS, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=config.N_LAYERS, help="Number of transformer layers")
    parser.add_argument("--block_size", type=int, default=config.BLOCK_SIZE, help="Context block size")
    parser.add_argument("--dropout", type=float, default=config.DROPOUT, help="Dropout probability")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Base learning rate")
    args = parser.parse_args()

    torch.manual_seed(config.SEED)
    
    # Device detection
    if args.device:
        device = args.device
    else:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available()
                  else "cpu")
    print(f"Using device: {device}")

    # Set up AMP context and GradScaler
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

    # --- Load text / setup vocabulary ---
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
            from toy_gpt.data import FALLBACK_TEXT
            raw_text = FALLBACK_TEXT

    # Check for checkpoints to resume
    checkpoint = None
    tokenizer_type = args.tokenizer
    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint_dir, "ckpt_latest.pt")
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            with open(checkpoint_path, 'rb') as f:
                checkpoint = torch.load(f, map_location=device)
            tokenizer_type = checkpoint.get('tokenizer_type', 'char')
            print(f"Restored tokenizer: '{tokenizer_type}' from checkpoint.")
        else:
            print(f"No checkpoint found at {checkpoint_path}. Starting training from scratch.")

    # Setup tokenization
    encode, decode, vocab_size, stoi, itos = get_tokenizer(
        tokenizer_type, 
        raw_text=raw_text, 
        checkpoint=checkpoint
    )

    # --- Build Model ---
    model = ToyGPT(
        vocab_size=vocab_size,
        n_embed=args.n_embed,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        block_size=args.block_size,
        dropout=args.dropout
    ).to(device)

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
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr)

    # Load weights if resuming
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            print(f"Warning: Could not restore optimizer state: {e}. Starting optimizer from scratch.")
        start_iter = checkpoint['iter_num'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from step {start_iter} with best validation loss {best_val_loss:.4f}")

    # Compile model if requested
    if args.compile:
        print("Compiling the model... (this may take a minute)")
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.2f}M")

    # --- Encode Data ---
    encoded_data = encode(raw_text)
    print(f"Corpus: {len(raw_text)} chars, encoded {len(encoded_data)} tokens, vocab size: {vocab_size}")

    data = torch.tensor(encoded_data, dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - args.block_size, (args.batch_size,))
        x = torch.stack([d[i:i + args.block_size] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + args.block_size] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss(model):
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(config.EVAL_ITERS)
            for k in range(config.EVAL_ITERS):
                xb, yb = get_batch(split)
                with ctx:
                    _, loss = model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    # Cosine learning rate decay scheduler with warmup
    def get_lr(it):
        if it < config.WARMUP_ITERS:
            return args.lr * it / config.WARMUP_ITERS
        if it > args.max_iters:
            return config.MIN_LR
        decay_ratio = (it - config.WARMUP_ITERS) / (args.max_iters - config.WARMUP_ITERS)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config.MIN_LR + coeff * (args.lr - config.MIN_LR)

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
            'config': {
                'n_embed': args.n_embed,
                'n_heads': args.n_heads,
                'n_layers': args.n_layers,
                'block_size': args.block_size,
            }
        }
        if tokenizer_type == "char":
            checkpoint_data['stoi'] = stoi
            checkpoint_data['itos'] = itos
            
        # Save to a temporary file first and then replace to avoid locked file errors on Windows
        temp_path = ckpt_path + ".tmp"
        torch.save(checkpoint_data, temp_path)
        os.replace(temp_path, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    # --- Train ---
    print(f"Starting/resuming training from step {start_iter} to {args.max_iters}...")
    for it in range(start_iter, args.max_iters + 1):
        # Set learning rate
        lr = get_lr(it) if config.DECAY_LR else args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if it % config.EVAL_INTERVAL == 0 or it == args.max_iters:
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

    # --- Sample text ---
    print("\n--- sample ---")
    if tokenizer_type == "char":
        start_char = '\n' if '\n' in stoi else list(stoi.keys())[0]
        start = torch.tensor([[stoi[start_char]]], dtype=torch.long, device=device)
    else:
        start = torch.tensor([[13]], dtype=torch.long, device=device) # newline in BPE
        
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    out = raw_model.generate(start, max_new_tokens=100)[0].tolist()
    print(decode(out))

if __name__ == "__main__":
    main()
