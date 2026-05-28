import argparse
import os
import sys
import torch

from toy_gpt import ToyGPT, get_tokenizer, config

def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained Toy GPT checkpoint.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory containing checkpoints.")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Specific path to a checkpoint file. Overrides --checkpoint_dir.")
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt to prime the generation.")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of tokens to generate.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use: 'cuda', 'mps', or 'cpu'. Defaults to auto-detect.")
    args = parser.parse_args()

    # Device detection
    if args.device:
        device = args.device
    else:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available()
                  else "cpu")
    print(f"Using device: {device}")

    # Determine checkpoint file path
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        possible_paths = [
            os.path.join(args.checkpoint_dir, "ckpt_best.pt"),
            os.path.join(args.checkpoint_dir, "ckpt_latest.pt")
        ]
        checkpoint_path = possible_paths[0] if os.path.exists(possible_paths[0]) else possible_paths[1]
        
    if not os.path.exists(checkpoint_path):
        print(f"Error: Could not find checkpoint file at {checkpoint_path}")
        sys.exit(1)

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Restore tokenizer and vocabulary size
    tokenizer_type = checkpoint.get('tokenizer_type', 'char')
    encode, decode, vocab_size, stoi, itos = get_tokenizer(
        tokenizer_type, 
        checkpoint=checkpoint
    )

    # Restore model dimensions
    model_config = checkpoint.get('config', {})
    n_embed = model_config.get('n_embed', config.N_EMBED)
    n_heads = model_config.get('n_heads', config.N_HEADS)
    n_layers = model_config.get('n_layers', config.N_LAYERS)
    block_size = model_config.get('block_size', config.BLOCK_SIZE)

    # Reconstruct Model
    model = ToyGPT(
        vocab_size=vocab_size,
        n_embed=n_embed,
        n_heads=n_heads,
        n_layers=n_layers,
        block_size=block_size,
        dropout=0.0 # disable dropout for inference
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Resumed from checkpoint at step {checkpoint['iter_num']} with best loss {checkpoint.get('best_val_loss', float('inf')):.4f}")

    # --- Sample text ---
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
            
    out = model.generate(start, max_new_tokens=args.num_samples)[0].tolist()
    print(decode(out))

if __name__ == "__main__":
    main()
