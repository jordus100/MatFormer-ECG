import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Encoder Block
# ---------------------------------------------------------------------------

class ECGformerEncoderBlock(nn.Module):
    """
    Pre-LN Transformer encoder block (Fig. 2 in paper):
      LayerNorm → MultiHeadSelfAttention → residual
      LayerNorm → FFN (d_model→128→64→d_model) → residual
    """

    def __init__(self, d_model: int, num_heads: int, mlp_units: list, dropout: float):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        layers = []
        in_dim = d_model
        for units in mlp_units:
            layers += [nn.Linear(in_dim, units), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = units
        layers += [nn.Linear(in_dim, d_model), nn.Dropout(dropout)]
        self.ffn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.attn_drop(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# ECGformer
# ---------------------------------------------------------------------------

class ECGformer(nn.Module):
    """
    Encoder-only Transformer for ECG arrhythmia classification.

    Default args reproduce the paper's 36,301 parameter count.
    Adjust input_length and num_classes for your dataset.
    """

    def __init__(
        self,
        input_length: int = 187,
        patch_size: int = 11,
        d_model: int = 128,
        num_heads: int = 8,        # must divide d_model; 2 works for d_model=4
        num_layers: int = 4,
        mlp_units: list = None,
        dropout: float = 0.15,
        num_classes: int = 5,
    ):
        super().__init__()
        if mlp_units is None:
            mlp_units = [128, 64]

        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.input_length = input_length

        self.patch_size = patch_size
        num_patches = input_length // patch_size  # e.g. 187 // 11 = 17 patches
        self.input_proj = nn.Linear(patch_size, d_model)
        self.pos_embedding = nn.Embedding(num_patches, d_model)

        # Encoder stack
        self.encoder_blocks = nn.ModuleList([
            ECGformerEncoderBlock(d_model, num_heads, mlp_units, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Classifier
        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_length) — zero-mean normalised ECG beat
        Returns:
            logits: (batch, num_classes)
        """
        B, L = x.shape

        # Step 1: trim the signal so it divides evenly into patches
        # e.g. 187 timesteps with patch_size=11 → keep 187 timesteps,
        # but 187//11=17 patches of 11 = 187, so no trimming needed here.
        # if patch_size=10: 187//10=18 patches → keep only 180 timesteps
        num_patches = L // self.patch_size
        x = x[:, :num_patches * self.patch_size]
        # x is now (32, 176) if patch_size=11, or (32, 180) if patch_size=10

        # Step 2: reshape into patches
        # view() splits the length dimension into (num_patches, patch_size)
        x = x.view(B, num_patches, self.patch_size)
        # x is now (32, 17, 11) — 17 patches each containing 11 timesteps

        # Step 3: project each patch to d_model
        # input_proj is Linear(patch_size, d_model) = Linear(11, 128)
        # it operates on the last dimension, so each patch vector of size 11
        # becomes a token vector of size 128
        x = self.input_proj(x)
        # x is now (32, 17, 128) — 17 tokens each of size d_model

        # Step 4: add positional embeddings (one per patch, not per timestep)
        positions = torch.arange(num_patches, device=x.device)
        x = x + self.pos_embedding(positions)
        # x is still (32, 17, 128)

        # Encoder blocks
        for block in self.encoder_blocks:
            x = block(x)

        x = self.norm(x)

        # Global average pooling → (B, d_model)
        x = x.mean(dim=1)

        return self.classifier(x)                      # (B, num_classes)


def build_model(
    input_length: int = 187,
    num_classes: int = 5,
    patch_size: int = 17,
    device: str = "cpu",
) -> ECGformer:
    model = ECGformer(
        input_length=input_length,
        patch_size=patch_size,
        d_model=128,
        num_heads=8,
        num_layers=4,
        mlp_units=[128, 64],
        dropout=0.15,
        num_classes=num_classes,
    )
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ECGformer | trainable params: {total:,}")
    return model.to(device)


def build_criterion(y_train_numpy, device: str) -> nn.CrossEntropyLoss:
    """Weighted cross-entropy to handle class imbalance."""
    import numpy as np
    counts = np.bincount(y_train_numpy)
    weights = torch.tensor(1.0 / counts, dtype=torch.float32)
    weights = weights / weights.sum()
    return nn.CrossEntropyLoss(weight=weights.to(device))


def build_optimizer(model: ECGformer) -> torch.optim.Adam:
    return torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-7)


def train_one_epoch(model, loader, optimizer, scheduler, device, criterion) -> float:
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, dtype=torch.float32)
        y_batch = y_batch.to(device, dtype=torch.long)
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * x_batch.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> dict:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, dtype=torch.float32)
        y_batch = y_batch.to(device, dtype=torch.long)
        logits = model(x_batch)
        total_loss += criterion(logits, y_batch).item() * x_batch.size(0)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += x_batch.size(0)
    return {"loss": total_loss / total, "accuracy": correct / total}
