from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_cifar10_datasets
from models import CNNBaseline, build_resnet18


EXPERIMENTS = {
    "E1": {"model": "cnn", "dropout": 0.0, "lr": 1e-3},
    "E2": {"model": "cnn", "dropout": 0.5, "lr": 1e-3},
    "E3": {"model": "resnet18", "dropout": 0.0, "lr": 1e-4},
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config):
    if config["model"] == "cnn":
        return CNNBaseline(dropout=config["dropout"])
    if config["model"] == "resnet18":
        return build_resnet18(pretrained=True)
    raise ValueError(f"Model tidak dikenal: {config['model']}")


def run_epoch(model, loader, criterion, optimizer, device, training=True):
    model.train(training)
    losses, y_true, y_pred = [], [], []
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if training:
            loss.backward()
            optimizer.step()
        losses.append(loss.item() * images.size(0))
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
    total = len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "loss": float(sum(losses) / total),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_one(experiment: str, epochs: int, batch_size: int, output_dir: Path, device: str):
    config = EXPERIMENTS[experiment]
    set_seed(42)
    train_ds, val_ds, test_ds = get_cifar10_datasets()
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0),
    }

    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    history = []
    best_val = float("inf")
    best_path = output_dir / f"{experiment}_best.pt"
    patience, bad_epochs = 5, 0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], criterion, optimizer, device, True)
        val_metrics = run_epoch(model, loaders["val"], criterion, optimizer, device, False)
        history.append({
            "experiment": experiment,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        })
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = run_epoch(model, loaders["test"], criterion, optimizer, device, False)
    pd.DataFrame(history).to_csv(output_dir / f"{experiment}_history.csv", index=False)
    with open(output_dir / f"{experiment}_metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in test_metrics.items() if k not in {"y_true", "y_pred"}}, f, indent=2)
    pd.DataFrame({"y_true": test_metrics["y_true"], "y_pred": test_metrics["y_pred"]}).to_csv(
        output_dir / f"{experiment}_predictions.csv", index=False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=list(EXPERIMENTS) + ["all"], default="E1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiments = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    for experiment in experiments:
        print(f"\n=== Menjalankan {experiment}: {EXPERIMENTS[experiment]} ===")
        train_one(experiment, args.epochs, args.batch_size, args.output_dir, args.device)


if __name__ == "__main__":
    main()
