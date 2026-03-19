import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

import ecgformer
import evaluate

EPOCHS      = 400
DEVICE      = "cuda"
BATCH_SIZE  = 32
DATA_X      = "/home/matrioszka/mit-bih/mitbih_beats_x.npy"
DATA_Y      = "/home/matrioszka/mit-bih/mitbih_beats_y.npy"
savepath = f"data/ecgmatformer_{datetime.now().strftime('%m-%d-%H:%M')}"

MODEL_KWARGS = dict(
    input_length    = 150,
    patch_size      = 10,
    d_model         = 128,
    num_heads       = 8,
    num_layers      = 4,
    d_ffn           = 128,
    dropout         = 0.15,
    num_classes     = 2,
    device          = DEVICE,
    matryoshka_depth= 4,
)

def load_data() -> tuple[DataLoader, DataLoader, LabelEncoder]:
    beats_x = np.load(DATA_X, allow_pickle=True)
    beats_y = np.load(DATA_Y, allow_pickle=True)
    print(f"Loaded data: X={beats_x.shape}  y={beats_y.shape}")

    beats_y = np.array(["N" if label == "N" else "O" for label in beats_y])

    bx_tr, bx_te, by_tr, by_te = train_test_split(
        beats_x, beats_y, test_size=0.2, random_state=45, stratify=beats_y
    )

    le = LabelEncoder()
    y_train = le.fit_transform(by_tr)
    y_test  = le.transform(by_te)
    print(f"Classes: {le.classes_}")

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(bx_tr, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(bx_te, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    return train_loader, test_loader, le

def run_training(savepath: str):
    train_loader, test_loader, le = load_data()

    # ── class-weighted loss ──
    y_train_all = train_loader.dataset.tensors[1].numpy()
    class_counts  = np.bincount(y_train_all)
    class_weights = 1.0 / class_counts
    class_weights /= class_weights.sum()
    print(f"Class weights: {class_weights}")
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    )

    model = ecgformer.ECGMatformer(**MODEL_KWARGS).to(DEVICE)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ECGMatformer | trainable params: {total:,}")

    optimizer = ecgformer.build_optimizer(model)

    best_val_loss  = float("inf")
    train_results  = []
    val_results    = []

    for epoch in range(1, EPOCHS):
        train_results.append(
            ecgformer.train_one_epoch(model, train_loader, optimizer, DEVICE, criterion)
        )
        val_results.append(
            ecgformer.evaluate(model, test_loader, DEVICE, criterion)
        )
        print(
            f"\nEpoch [{epoch:3d}/{EPOCHS}]\n"
            f"  train: {train_results[-1]}\n"
            f"  val:   {val_results[-1]}"
        )

        current_val_loss = val_results[-1][0]["val_loss"]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(model.state_dict(), savepath + ".pth")
            print(f"  ✓ saved best model → {savepath}.pth")

        # persist results after every epoch
        with open(savepath + "_train.pickle", "wb") as f:
            pickle.dump(train_results, f)
        with open(savepath + "_validation.pickle", "wb") as f:
            pickle.dump(val_results, f)

    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss":                 best_val_loss,
        },
        savepath + ".tar",
    )
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    return model, test_loader, le


def run_evaluation(
    name: str,
    model: ecgformer.ECGMatformer | None,
    test_loader: DataLoader | None,
    show: bool,
):
    # load data + model if coming from --mode eval
    if model is None or test_loader is None:
        _, test_loader, le = load_data()
        model = ecgformer.ECGMatformer(**MODEL_KWARGS).to(DEVICE)
        pth = Path(f"{name}.pth")
        if not pth.exists():
            print(f"ERROR: model weights not found: {pth}", file=sys.stderr)
            sys.exit(1)
        model.load_state_dict(torch.load(pth, map_location=DEVICE))
        print(f"Loaded weights from {pth}")
    else:
        le = None   # label names not available when coming straight from training

    model.eval()
    granularities = list(range(model.matryoshka_depth))[::-1]

    # ── collect preds per granularity ──
    results: dict[int, dict] = {}
    with torch.no_grad():
        for g in granularities:
            all_preds, all_labels = [], []
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch, [g for _ in range(len(model.encoder_blocks))])
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

            target_names = le.classes_ if le is not None else None
            report = classification_report(
                all_labels, all_preds,
                target_names=target_names,
                output_dict=True,
            )
            cm = confusion_matrix(all_labels, all_preds)
            results[g] = {"report": report, "cm": cm, "labels": all_labels, "preds": all_preds}
            print(f"\nMatryoshka Granularity: {g}")
            print(classification_report(all_labels, all_preds, target_names=target_names))

    evaluate.plot_evaluation(name, results, show)

def main():
    savepath = str(output_dir / f"ecgmatformer_{datetime.now().strftime('%b-%d-%H-%M')}")
    model, test_loader, _ = run_training(
        savepath     = savepath,
    )

if __name__ == "__main__":
    main()