import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from toy_gpt import config

class CausalSelfAttention(nn.Module):
    """A standard multi-head causal self-attention layer with support for
    scaled dot-product attention (PyTorch 2.0+)."""
    def __init__(self, n_embed=config.N_EMBED, n_heads=config.N_HEADS, dropout=config.DROPOUT):
        super().__init__()
        assert n_embed % n_heads == 0
        self.c_attn = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.c_proj = nn.Linear(n_embed, n_embed)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.n_embed = n_embed

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)
        
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.attn_dropout.p if self.training else 0.0, 
            is_causal=True
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class FeedForward(nn.Module):
    """A small position-wise MLP."""
    def __init__(self, n_embed=config.N_EMBED, dropout=config.DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """A transformer block: attention then MLP, each with a residual + layernorm."""
    def __init__(self, n_embed=config.N_EMBED, n_heads=config.N_HEADS, dropout=config.DROPOUT):
        super().__init__()
        self.sa   = CausalSelfAttention(n_embed, n_heads, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1  = nn.LayerNorm(n_embed)
        self.ln2  = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class ToyGPT(nn.Module):
    """The full GPT language model."""
    def __init__(self, vocab_size, n_embed=config.N_EMBED, n_heads=config.N_HEADS, 
                 n_layers=config.N_LAYERS, block_size=config.BLOCK_SIZE, dropout=config.DROPOUT):
        super().__init__()
        self.block_size = block_size
        self.token_embedding    = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.blocks  = nn.Sequential(*[Block(n_embed, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f    = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
        # Tie input embeddings and language model head weights
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
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Autoregressively sample max_new_tokens tokens."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]             # focus on the last step
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
