import os
import urllib.request

# Default fallback text
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
) * 200

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

def get_tokenizer(tokenizer_type, raw_text=None, checkpoint=None):
    """
    Returns encode, decode, vocab_size, stoi, itos for the given tokenizer_type.
    """
    if tokenizer_type == "gpt2":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
        vocab_size = 50257
        stoi, itos = None, None
        print(f"Using GPT-2 BPE tokenizer. Vocab size: {vocab_size}")
    else:
        # Character-level tokenizer
        if checkpoint is not None:
            stoi = checkpoint['stoi']
            itos = checkpoint['itos']
            vocab_size = checkpoint['vocab_size']
        else:
            if raw_text is None:
                raw_text = FALLBACK_TEXT
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
        
    return encode, decode, vocab_size, stoi, itos
