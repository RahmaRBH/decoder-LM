
# Decoder-only transformer language model
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import tiktoken
import wandb

# --- Config ---
@dataclass
class GPTConfig:
    batch_size:    int   = 64
    block_size:    int   = 32
    max_iters:     int   = 5000
    eval_interval: int   = 500
    learning_rate: float = 3e-4
    device:        str   = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters:    int   = 200
    n_embd:        int   = 384
    n_layer:       int   = 6
    n_head:        int   = 6
    dropout:       float = 0.2
    data_dir:      str   = '.'          # directory containing train.bin and val.bin
    resume:        bool  = False         # set True to load from the latest checkpoint

cfg = GPTConfig()

torch.manual_seed(1337)

# --- BPE tokenization (GPT-2 vocabulary, 50257 tokens) ---
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
decode = lambda l: enc.decode(l)

# --- Memory-mapped data loading (train.bin / val.bin built by prepare_data.py) ---
# np.memmap reads directly from disk — no full corpus in RAM at startup
train_data = np.memmap(os.path.join(cfg.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data   = np.memmap(os.path.join(cfg.data_dir, 'val.bin'),   dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix   = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
    # cast uint16 → int64 here; doing it per-slice keeps peak memory low
    x = torch.stack([torch.from_numpy(data[i:i+cfg.block_size].astype(np.int64))   for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+cfg.block_size+1].astype(np.int64)) for i in ix])
    x, y = x.to(cfg.device), y.to(cfg.device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        # lower-triangular mask makes this a decoder (no peeking at future tokens)
        self.register_buffer('tril', torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # GELU matches GPT-2 (smoother gradient flow than ReLU)
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks  = nn.Sequential(*[Block(cfg.n_embd, n_head=cfg.n_head) for _ in range(cfg.n_layer)])
        self.ln_f    = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos_emb   = self.position_embedding_table(torch.arange(T, device=cfg.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -cfg.block_size:]
            logits, _ = self(idx_cond)
            logits    = logits[:, -1, :]
            probs     = F.softmax(logits, dim=-1)
            idx_next  = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# --- W&B: run name encodes key hyperparams for easy comparison across runs ---
run_name = f"gpt_embd{cfg.n_embd}_layer{cfg.n_layer}_head{cfg.n_head}"
wandb.init(project="nanoGPT", name=run_name, config=vars(cfg))

model = BigramLanguageModel().to(cfg.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

# --- Checkpoint: resume from latest .pt file in checkpoints/ if cfg.resume ---
os.makedirs("checkpoints", exist_ok=True)
best_val_loss = float('inf')
start_iter    = 0

if cfg.resume:
    ckpts = sorted(f for f in os.listdir("checkpoints") if f.endswith(".pt"))
    if ckpts:
        ckpt_path = os.path.join("checkpoints", ckpts[-1])
        ckpt = torch.load(ckpt_path, map_location=cfg.device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_iter    = ckpt['iter']
        best_val_loss = ckpt['best_val_loss']
        print(f"Resumed from {ckpt_path} at iter {start_iter}")

# --- Training loop ---
for iter in range(start_iter, cfg.max_iters):

    if iter % cfg.eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        wandb.log({
            "iter":       iter,
            "train/loss": losses['train'],
            "val/loss":   losses['val'],
            "lr":         cfg.learning_rate,
        })

        # Save checkpoint whenever val loss improves
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            ckpt = {
                'model':         model.state_dict(),
                'optimizer':     optimizer.state_dict(),
                'config':        vars(cfg),
                'iter':          iter,
                'best_val_loss': best_val_loss,
            }
            ckpt_file = os.path.join("checkpoints", f"ckpt_iter{iter:05d}.pt")
            torch.save(ckpt, ckpt_file)
            print(f"Checkpoint saved → {ckpt_file}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

wandb.finish()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
