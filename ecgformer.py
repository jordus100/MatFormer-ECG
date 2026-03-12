import torch
import torch.nn as nn
import math

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
      LayerNorm → FFN (d_model→[mlp_units]→d_model) → residual
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

class ECGformer(nn.Module):

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
        device: str = "cuda",
    ):
        super().__init__()
        self.d_model = d_model
        if mlp_units is None:
            mlp_units = [128, 64]

        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.input_length = input_length
        self.pos_encoding = positionalencoding1d(d_model, input_length // patch_size).to(device)

        self.patch_size = patch_size
        num_patches = input_length // patch_size  # e.g. 187 // 11 = 17 patches
        self.input_proj = nn.Linear(patch_size, d_model)

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
        for block in self.encoder_blocks:
            x = block(x)
        x = self.norm(x)
        # print(f"Transformer output shape: {x.shape}")
        # x.shape = (B, num_patches, d_model)
        # Global average pooling → (B, d_model)
        x = x.mean(dim=1)
        return self.classifier(x) # (B, num_classes)

def build_model(
    input_length: int = 187,
    num_classes: int = 5,
    patch_size: int = 17,
    num_heads: int = 8,
    mlp_units: list = [128, 64],
    d_model: int = 128,
    device: str = "cuda",
) -> ECGformer:
    model = ECGformer(
        input_length=input_length,
        patch_size=patch_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=4,
        mlp_units=mlp_units,
        dropout=0.15,
        num_classes=num_classes,
        device=device,
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
