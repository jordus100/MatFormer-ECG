import torch
import torch.nn as nn
import math
import matformer

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

class ECGformerEncoderBlock(nn.Module):
    """
    Pre-LN Transformer encoder block (Fig. 2 in paper):
      LayerNorm → MultiHeadSelfAttention → residual
      LayerNorm → FFN (d_model→d_ffn→d_model) → residual
    """
    def __init__(self, d_model: int, num_heads: int, d_ffn, dropout: float, matryoshka_depth: int = 3, device="cuda"):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = matformer.MultiHeadSelfAttentionMatryoshka(
            E_q=d_model,
            E_total=d_model,
            nheads=num_heads,
            matryoshka_depth=matryoshka_depth,
            dropout=dropout,
            device=device
        )
        self.attn_drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.ModuleList([matformer.LinearMatryoshka(d_model, d_ffn, matryoshka_depth, 1, device=device), nn.ReLU(), nn.Dropout(dropout),
        matformer.LinearMatryoshka(d_ffn, d_model, matryoshka_depth, 0, device=device), nn.Dropout(dropout)])

    def forward(self, x: torch.Tensor, matryoshka_granularity) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out = self.attn(normed, matryoshka_granularity)
        x = x + self.attn_drop(attn_out)
        x = self.norm2(x)
        for layer in self.ffn:
            x = layer(x, matryoshka_granularity) if isinstance(layer, matformer.LinearMatryoshka) else layer(x)
        return x

class ECGMatformer(nn.Module):

    def __init__(
        self,
        input_length: int = 187,
        patch_size: int = 11,
        d_model: int = 128,
        num_heads: int = 8,        # must divide d_model; 2 works for d_model=4
        num_layers: int = 4,
        d_ffn: int = 128,
        dropout: float = 0.15,
        num_classes: int = 5,
        device: str = "cuda",
        matryoshka_depth: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.matryoshka_depth = matryoshka_depth

        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.input_length = input_length
        self.pos_encoding = positionalencoding1d(d_model, input_length // patch_size).to(device)

        self.patch_size = patch_size
        num_patches = input_length // patch_size  # e.g. 187 // 11 = 17 patches
        self.input_proj = nn.Linear(patch_size, d_model)

        # Encoder stack
        self.encoder_blocks = nn.ModuleList([
            ECGformerEncoderBlock(d_model, num_heads, d_ffn, dropout, matryoshka_depth, device="cuda")
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Classifier
        self.classifier1 = nn.Linear(d_model, d_model // 2)
        self.classifier2 = nn.Linear(d_model // 2, num_classes)

    def forward(self, x: torch.Tensor, matryoshka_granularity) -> torch.Tensor:
        """
        x: (batch, input_length) — zero-mean normalised ECG beat
        """
        B, L = x.shape
        num_patches = L // self.patch_size
        x = x[:, :num_patches * self.patch_size]
        x = x.view(B, num_patches, self.patch_size)
        x = self.input_proj(x)
        # print("Input projection output shape: ", x.shape)
        # x.shape = (B, num_patches, d_model)
        x = x + self.pos_encoding
        cls = torch.empty(B, 1, self.d_model).to(x.device)
        nn.init.normal_(cls)
        x = torch.cat((cls, x), 1)
        for block in self.encoder_blocks:
            x = block(x, matryoshka_granularity)
        x = self.norm(x)
        # print(f"Transformer output shape: {x.shape}")
        # x.shape = (B, num_patches, d_model)
        # Global average pooling → (B, d_model)
        # x = x.mean(dim=1)
        x = self.classifier1(x[:, 0, :]) # (B, num_classes)
        return self.classifier2(x)

def build_criterion(y_train_numpy, device: str) -> nn.CrossEntropyLoss:
    """Weighted cross-entropy to handle class imbalance."""
    import numpy as np
    counts = np.bincount(y_train_numpy)
    weights = torch.tensor(1.0 / counts, dtype=torch.float32)
    weights = weights / weights.sum()
    return nn.CrossEntropyLoss(weight=weights.to(device), reduction='sum')

def build_optimizer(model: ECGMatformer) -> torch.optim.Adam:
    return torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-7)

def train_one_epoch(model, loader, optimizer, device, criterion, scheduler=None) -> list[dict]:
    model.train()
    mat_grans = [i for i in range(model.matryoshka_depth)][::-1]
    loss = dict()
    for gran in mat_grans:
        loss[gran] = {"sum_loss": 0.0, "train_count": 0}
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, dtype=torch.float32)
        y_batch = y_batch.to(device, dtype=torch.long)
        optimizer.zero_grad()
        mat_gran = torch.randint(0, model.matryoshka_depth, (1,)).item()
        loss = criterion(model(x_batch, mat_gran), y_batch)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss[mat_gran]["sum_loss"] += loss.item()
        loss[mat_gran]["train_count"] += x_batch.size[0]
    results = []
    for gran in mat_grans:
        results.append({"mat_gran": gran, "train_loss": loss[gran]["sum_loss"] / (loss[gran]["train_count"])})
    return results

@torch.no_grad()
def evaluate(model, loader, device, criterion) -> dict:
    model.eval()
    matryoshka_granularities = [i for i in range(model.matryoshka_depth)][::-1]
    total_loss, correct, total = 0.0, 0, 0
    for g in matryoshka_granularities:
        total_loss, correct, total = 0.0, 0, 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, dtype=torch.float32)
            y_batch = y_batch.to(device, dtype=torch.long)
            logits = model(x_batch, g)
            total_loss += criterion(logits, y_batch).item() * x_batch.size(0)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += x_batch.size(0)
        print({"matryoshka_granularity": g, "loss": total_loss / total, "accuracy": correct / total})

    return {"loss": total_loss / total, "accuracy": correct / total}
