import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
import pickle

import ecgformer

epochs = 100
savepath = f"ecgmatformer_{datetime.now().strftime('%b-%d-%H-%M')}"

beats_x = np.load('/home/matrioszka/mit-bih/mitbih_beats_x.npy', allow_pickle=True)
beats_y = np.load('/home/matrioszka/mit-bih/mitbih_beats_y.npy', allow_pickle=True)
print(beats_x.shape, beats_y.shape)

beats_y = np.array(['N' if label=='N' else 'O' for label in beats_y])
# split into train/test
beats_train_x, beats_test_x, beats_train_y, beats_test_y = train_test_split(
    beats_x, beats_y, test_size=0.2, random_state=45, stratify=beats_y)

le = LabelEncoder()
y_train = le.fit_transform(beats_train_y)
y_test  = le.transform(beats_test_y)

print(le.classes_)

X_train_t = torch.tensor(beats_train_x, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)

X_test_t = torch.tensor(beats_test_x, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

class_counts = np.bincount(y_train)  # assumes integer-encoded labels 0–4
print(class_counts)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()  # normalize to sum to 1
print(class_weights)

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to("cuda")
criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

model = ecgformer.ECGMatformer(
    input_length=150,
    patch_size=10,
    d_model=128,
    num_heads=8,
    num_layers=4,
    d_ffn=128,
    dropout=0.15,
    num_classes=2,
    device="cuda",
    matryoshka_depth=4,
)
total = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"ECGformer | trainable params: {total:,}")
model = model.to("cuda")
optimizer = ecgformer.build_optimizer(model)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer, max_lr=1e-3,
#     steps_per_epoch=len(train_loader),
#     epochs=100,
#     pct_start=0.1,
# )

best_val_loss = float("inf")
train_results = []
val_results = []

for epoch in range(1, epochs):
    train_results.append(ecgformer.train_one_epoch(model, train_loader, optimizer, "cuda", criterion))
    val_results.append(ecgformer.evaluate(model, test_loader, "cuda", criterion))
    # scheduler.step(val_loss)
    print(
        f"\nEpoch [{epoch:3d}/{epochs}] "
    )
    print("Training results:")
    print(train_results[-1])
    print("Validation results:")
    print(val_results[-1])

    if val_results[-1][0]["val_loss"] < best_val_loss:
        best_val_loss = val_results[-1][0]["val_loss"]
        torch.save(model.state_dict(), savepath+".pth")
        print(f"\nSaved best model → {savepath+'.pth'}")
    train_file = open(savepath+"_train.pickle", "wb")
    val_file = open(savepath+"_validation.pickle", "wb")
    pickle.dump(train_results, train_file)
    pickle.dump(val_results, val_file)

torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss,
            }, savepath+".tar")


print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
