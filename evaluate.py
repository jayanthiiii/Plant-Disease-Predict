"""
evaluate.py — Evaluate a trained model and generate classification report.

Usage
-----
python evaluate.py --model EfficientNetB0 --data_dir data/plantvillage --weights weights/efficientnetb0_plantvillage.h5
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PlantGuard AI — Evaluation")
    p.add_argument("--model",    default="EfficientNetB0",
                   choices=["EfficientNetB0", "MobileNetV3"])
    p.add_argument("--data_dir", default="data/plantvillage")
    p.add_argument("--weights",  default=None, help="Path to .h5 weights file.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--img_size",   type=int, default=224)
    p.add_argument("--output_dir", default="results")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    IMG = (args.img_size, args.img_size)

    ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        image_size=IMG,
        batch_size=args.batch_size,
        label_mode="categorical",
        shuffle=False,
    )
    class_names = ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=AUTOTUNE,
    ).prefetch(AUTOTUNE)

    # ── Load model ────────────────────────────────────────────────────────────
    from models.predictor import DiseasePredictor
    predictor = DiseasePredictor(model_name=args.model)
    if args.weights and Path(args.weights).exists():
        predictor.model.load_weights(args.weights)
        logger.info("Loaded weights: %s", args.weights)

    # ── Collect predictions ───────────────────────────────────────────────────
    y_true, y_pred = [], []
    for images, labels in ds:
        preds = predictor.model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ── Classification report ─────────────────────────────────────────────────
    report = classification_report(y_true, y_pred, target_names=class_names)
    logger.info("\n%s", report)
    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("Saved report → %s", report_path)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(f"{args.model} — Confusion Matrix", fontsize=16)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    logger.info("Saved confusion matrix → %s", cm_path)

    # ── Accuracy summary ──────────────────────────────────────────────────────
    accuracy = (y_true == y_pred).mean()
    logger.info("Overall accuracy: %.4f", accuracy)


if __name__ == "__main__":
    main()
