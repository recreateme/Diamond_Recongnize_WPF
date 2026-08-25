#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对照实验：训练一致预处理 (A) vs 现状 Resize+16+CenterCrop (B)。

同一 ONNX/PT 模型、同一批图，只换 transform，对比：
  · 有标签目录（ImageFolder：类别子目录）时：Accuracy / Macro-F1 / 逐类 F1
  · 无标签时：预测一致性（A vs B 同预测比例、平均置信度差）

用法示例：
  python scripts/compare_preprocess_ab.py ^
    --onnx checkpoints/model.onnx ^
    --data_dir outputs/acceptance_sahi/dog/crops
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "python_core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from inference_common import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    read_class_thresholds,
    read_model_meta,
    softmax,
    decide_class,
    build_threshold_vector,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_images(data_dir: Path) -> List[Tuple[Path, Optional[str]]]:
    """返回 (path, label_or_None)。有类别子目录则视为有标签。"""
    items: List[Tuple[Path, Optional[str]]] = []
    subdirs = [d for d in sorted(data_dir.iterdir()) if d.is_dir()]
    labeled = False
    if subdirs:
        for d in subdirs:
            files = [p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS]
            if files:
                labeled = True
                for p in files:
                    items.append((p, d.name))
    if not labeled:
        for p in sorted(data_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                items.append((p, None))
    return items


def preprocess(img: Image.Image, img_size: int, mode: str) -> np.ndarray:
    """mode: 'A' = Resize only；'B' = Resize+16 + CenterCrop。"""
    rgb = img.convert("RGB")
    if mode == "A":
        rgb = rgb.resize((img_size, img_size), Image.BILINEAR)
    else:
        side = img_size + 16
        rgb = rgb.resize((side, side), Image.BILINEAR)
        off = (side - img_size) // 2
        rgb = rgb.crop((off, off, off + img_size, off + img_size))
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN.reshape(1, 1, 3)) / IMAGENET_STD.reshape(1, 1, 3)
    return arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


def macro_f1(y_true: List[int], y_pred: List[int], n_classes: int) -> Tuple[float, List[float]]:
    per: List[float] = []
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per.append(f1)
    return float(np.mean(per)) if per else 0.0, per


def eval_mode(
    session,
    input_name: str,
    items: List[Tuple[Path, Optional[str]]],
    classes: List[str],
    img_size: int,
    thr_vec: Optional[np.ndarray],
    mode: str,
) -> Dict:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_true: List[int] = []
    y_pred: List[int] = []
    preds: List[str] = []
    confs: List[float] = []
    labeled = any(lab is not None for _, lab in items)

    for path, lab in items:
        with Image.open(path) as im:
            tensor = preprocess(im, img_size, mode)
        logits = session.run(None, {input_name: tensor})[0][0]
        scores = softmax(logits)
        idx = decide_class(scores, thr_vec)
        preds.append(classes[idx])
        confs.append(float(scores[idx]))
        if labeled and lab is not None and lab in class_to_idx:
            y_true.append(class_to_idx[lab])
            y_pred.append(idx)

    out: Dict = {
        "mode": mode,
        "n": len(items),
        "mean_confidence": float(np.mean(confs)) if confs else 0.0,
        "preds": preds,
        "confs": confs,
    }
    if y_true:
        acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
        mf1, per = macro_f1(y_true, y_pred, len(classes))
        out["accuracy"] = acc
        out["macro_f1"] = mf1
        out["per_class_f1"] = {classes[i]: per[i] for i in range(len(classes))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="预处理 A/B 对照（训练对齐 vs +16 crop）")
    ap.add_argument("--onnx", default=str(ROOT / "checkpoints" / "model.onnx"))
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "preprocess_ab_report.json"))
    args = ap.parse_args()

    import onnxruntime as ort

    onnx_path = Path(args.onnx)
    data_dir = Path(args.data_dir)
    items = collect_images(data_dir)
    if not items:
        print(f"[FAIL] 未找到图像: {data_dir}")
        return 1

    classes, img_size = read_model_meta(None, str(onnx_path))
    thr = read_class_thresholds(None, str(onnx_path))
    thr_vec = build_threshold_vector(classes, thr)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(str(onnx_path), so, providers=providers)
    except Exception:
        session = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    print(f"images={len(items)} img_size={img_size} classes={classes}")
    print(f"providers={session.get_providers()}")

    res_a = eval_mode(session, input_name, items, classes, img_size, thr_vec, "A")
    res_b = eval_mode(session, input_name, items, classes, img_size, thr_vec, "B")

    agree = sum(1 for a, b in zip(res_a["preds"], res_b["preds"]) if a == b)
    report = {
        "onnx": str(onnx_path),
        "data_dir": str(data_dir),
        "n": len(items),
        "labeled": "accuracy" in res_a,
        "A_train_aligned": {k: v for k, v in res_a.items() if k not in ("preds", "confs")},
        "B_legacy_crop": {k: v for k, v in res_b.items() if k not in ("preds", "confs")},
        "agreement_rate": agree / len(items),
        "disagree_count": len(items) - agree,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("--- A (Resize only, train-aligned) ---")
    print({k: v for k, v in res_a.items() if k not in ("preds", "confs")})
    print("--- B (Resize+16 + CenterCrop, legacy) ---")
    print({k: v for k, v in res_b.items() if k not in ("preds", "confs")})
    print(f"agreement A==B: {agree}/{len(items)} ({report['agreement_rate']:.1%})")
    print(f"report -> {out_path}")

    if report["labeled"]:
        better = "A" if res_a["macro_f1"] >= res_b["macro_f1"] else "B"
        print(f"macro_f1 winner: {better} (A={res_a['macro_f1']:.4f} B={res_b['macro_f1']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
