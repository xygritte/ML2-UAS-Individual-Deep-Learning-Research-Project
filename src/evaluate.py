from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


EXPERIMENTS = ["E1", "E2", "E3"]
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for exp in EXPERIMENTS:
        metrics_path = args.output_dir / f"{exp}_metrics.json"
        pred_path = args.output_dir / f"{exp}_predictions.csv"

        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            metrics["experiment"] = exp
            rows.append(metrics)

        if pred_path.exists():
            pred = pd.read_csv(pred_path)
            cm = confusion_matrix(
                pred["y_true"], pred["y_pred"], labels=list(range(10))
            )
            plt.figure(figsize=(8, 7))
            sns.heatmap(
                cm,
                annot=False,
                cmap="Blues",
                cbar=True,
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
            )
            plt.title(f"Confusion Matrix - {exp}")
            plt.xlabel("Prediksi")
            plt.ylabel("Label Asli")
            plt.tight_layout()
            plt.savefig(args.output_dir / f"{exp}_confusion_matrix.png", dpi=160)
            plt.close()

        history_path = args.output_dir / f"{exp}_history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path)
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(history["epoch"], history["train_loss"], label="Train")
            ax[0].plot(history["epoch"], history["val_loss"], label="Validation")
            ax[0].set_title(f"Loss - {exp}")
            ax[0].set_xlabel("Epoch")
            ax[0].legend()

            ax[1].plot(history["epoch"], history["train_accuracy"], label="Train")
            ax[1].plot(history["epoch"], history["val_accuracy"], label="Validation")
            ax[1].set_title(f"Accuracy - {exp}")
            ax[1].set_xlabel("Epoch")
            ax[1].legend()

            plt.tight_layout()
            fig.savefig(args.output_dir / f"{exp}_training_curves.png", dpi=160)
            plt.close(fig)

    if rows:
        summary = pd.DataFrame(rows)
        summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)
        print(summary.to_string(index=False))
    else:
        print("Belum ada hasil training. Jalankan src/train.py terlebih dahulu.")


if __name__ == "__main__":
    main()
