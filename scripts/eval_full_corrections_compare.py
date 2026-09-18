# -*- coding: utf-8 -*-
"""Compare production vs finetuned checkpoint on full corrections folder."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_core"))

from model_builder import build_model  # noqa: E402
from train import DefectDataset, get_transforms  # noqa: E402

FULL = Path(r"D:\迅雷下载\corrections")
OUT = ROOT / "checkpoints_ft_balanced_221" / "eval_full_corrections"
MODELS = {
    "prod_current": ROOT / "checkpoints" / "best_model.pt",
    "ft_balanced_221": ROOT / "checkpoints_ft_balanced_221" / "best_model.pt",
}


def load_ckpt(path: Path, classes: list[str], device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ck_classes = list(ckpt.get("classes") or classes)
    model = build_model(len(ck_classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device).eval()
    return model, ck_classes, float(ckpt.get("macro_f1") or 0)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_tf = get_transforms(256, pre_augmented=False)
    ds = DefectDataset([FULL], transform=val_tf)
    classes = ds.classes
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    print(f"eval set {FULL} n={len(ds)} classes={classes} device={device}")

    results: dict = {}
    for name, path in MODELS.items():
        print(f"\n== {name} {path}")
        model, ck_classes, hist_f1 = load_ckpt(path, classes, device)
        if list(ck_classes) != list(classes):
            raise RuntimeError(f"class mismatch: {ck_classes} vs {classes}")
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                logits = model(imgs.to(device))
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.numpy())
        y_true = np.asarray(all_labels)
        y_pred = np.asarray(all_preds)
        report = classification_report(
            y_true, y_pred, target_names=classes, digits=4, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)
        p, r, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(classes))), zero_division=0
        )
        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
        per_class = {
            classes[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(classes))
        }
        payload = {
            "model": name,
            "checkpoint": str(path),
            "ckpt_macro_f1_meta": hist_f1,
            "eval_dir": str(FULL),
            "n": int(len(y_true)),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "classes": classes,
            "report_text": report,
        }
        results[name] = payload
        (OUT / f"{name}_report.txt").write_text(
            report + "\n\nConfusion:\n" + str(cm) + "\n", encoding="utf-8"
        )
        print(report)
        print(f"Acc={acc:.4f} Macro-F1={macro_f1:.4f}")

    (OUT / "comparison.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 全量 corrections 评测对比",
        "",
        f"数据: `{FULL}`  n={len(ds)}",
        "",
        "| 模型 | Acc | Macro-F1 | Weighted-F1 |",
        "|---|---:|---:|---:|",
    ]
    for name, p in results.items():
        lines.append(
            f"| {name} | {p['accuracy']:.4f} | {p['macro_f1']:.4f} | {p['weighted_f1']:.4f} |"
        )
    for title, key in [("F1", "f1"), ("Recall", "recall"), ("Precision", "precision")]:
        lines += [
            "",
            f"## 逐类 {title}",
            "",
            "| 模型 | " + " | ".join(classes) + " |",
            "|---|" + "|".join(["---:"] * len(classes)) + "|",
        ]
        for name, p in results.items():
            cells = [f"{p['per_class'][c][key]:.4f}" for c in classes]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("saved", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
