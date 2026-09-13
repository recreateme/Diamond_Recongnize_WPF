#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理公共模块 — 开发版 (inference_engine) 与机台版 (inference_engine_onnx) 共用。

集中放置：
  · Letterbox（OpenCV，训练/推理一致）与 BGR 裁剪高速预处理
  · 元数据读取（类别 / img_size）
  · Softmax 与有效类别 argmax 决策（排除已废弃类别）
  · run_batch_predict / predict_bgr_crops_timed — 批量推理（可拆预处理/推理耗时）
  · logits_row_to_result / build_result_dict — 单张/批量统一结果结构
  · ImageNet 归一化常量

避免两处引擎逻辑漂移；修改预处理/阈值/批量逻辑时只改此处。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# 与 torchvision / train.py 一致的 ImageNet 统计量（HWC 布局，float32）
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGENET_MEAN_LIST = [0.485, 0.456, 0.406]
IMAGENET_STD_LIST = [0.229, 0.224, 0.225]

# 历史 5 类模型中已废弃的类别；真 3 类模型不含这些名，集合为空兼容
EXCLUDED_CLASSES = frozenset({"局部破损", "断钻"})

# SAHI 裁剪 letterbox 并行线程数（cv2 段可释放 GIL）
_PREPROCESS_WORKERS = max(1, min(8, (os.cpu_count() or 4)))


def active_classes(model_classes: List[str]) -> List[str]:
    """应用对外暴露的有效类别（排除 EXCLUDED_CLASSES，保持模型顺序）。"""
    return [c for c in model_classes if c not in EXCLUDED_CLASSES]


def active_class_indices(model_classes: List[str]) -> List[int]:
    """有效类别在 model_classes / scores 向量中的下标。"""
    return [i for i, c in enumerate(model_classes) if c not in EXCLUDED_CLASSES]


class LetterboxToSquare:
    """等比缩放后居中 pad 到 size×size（默认黑边），训练/验证/推理共用。

    几何用 OpenCV（与 letterbox_rgb_u8_to_chw 一致），输出仍为 PIL 以便接增强。
    """

    def __init__(self, size: int, fill: int = 0):
        self.size = int(size)
        self.fill = int(fill)

    def __call__(self, img):
        from PIL import Image as PILImage

        if getattr(img, "mode", None) != "RGB":
            img = img.convert("RGB")
        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        import cv2

        h, w = int(arr.shape[0]), int(arr.shape[1])
        if w <= 0 or h <= 0:
            raise ValueError(f"invalid image size: {w}x{h}")
        scale = min(self.size / w, self.size / h)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        resized = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.size, self.size, 3), self.fill, dtype=np.uint8)
        left = (self.size - nw) // 2
        top = (self.size - nh) // 2
        canvas[top : top + nh, left : left + nw] = resized
        return PILImage.fromarray(canvas)


def letterbox_rgb_u8_to_chw(
    rgb: np.ndarray,
    img_size: int,
    fill: int = 0,
) -> np.ndarray:
    """
    RGB uint8 HWC → CHW float32（ImageNet 归一化）。
    几何语义与 LetterboxToSquare 一致（等比缩放 + 居中黑边）。
    """
    import cv2

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expect HWC RGB, got shape={getattr(rgb, 'shape', None)}")
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid image size: {w}x{h}")
    size = int(img_size)
    scale = min(size / w, size / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), int(fill), dtype=np.uint8)
    left = (size - nw) // 2
    top = (size - nh) // 2
    canvas[top : top + nh, left : left + nw] = resized
    arr = canvas.astype(np.float32) * (1.0 / 255.0)
    arr = (arr - IMAGENET_MEAN.reshape(1, 1, 3)) / IMAGENET_STD.reshape(1, 1, 3)
    return arr.transpose(2, 0, 1)


def letterbox_bgr_u8_to_chw(
    bgr: np.ndarray,
    img_size: int,
    fill: int = 0,
) -> np.ndarray:
    """OpenCV BGR 裁剪 → CHW float32（内部转 RGB 后 letterbox）。"""
    import cv2

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return letterbox_rgb_u8_to_chw(rgb, img_size, fill)


def preprocess_bgr_crops_to_chw(
    crops_bgr: List[np.ndarray],
    img_size: int,
    *,
    workers: Optional[int] = None,
) -> Tuple[List[np.ndarray], float]:
    """
    批量 BGR 裁剪 → CHW 列表；返回 (tensors, preprocess_wall_s)。
    workers>1 时用线程池并行 letterbox。
    """
    n = len(crops_bgr)
    if n == 0:
        return [], 0.0
    n_workers = _PREPROCESS_WORKERS if workers is None else max(1, int(workers))
    t0 = time.perf_counter()
    if n_workers <= 1 or n < 4:
        tensors = [letterbox_bgr_u8_to_chw(c, img_size) for c in crops_bgr]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(n_workers, n)) as ex:
            tensors = list(
                ex.map(lambda c: letterbox_bgr_u8_to_chw(c, img_size), crops_bgr)
            )
    return tensors, time.perf_counter() - t0


def build_torchvision_eval_transform(img_size: int):
    """
    训练验证 / 推理共用的 eval 预处理（与 train.get_transforms 的 val_tf 一致）。

    Letterbox → ToTensor → ImageNet Normalize；禁止与推理分叉的 CenterCrop。
    """
    import torchvision.transforms as T

    return T.Compose([
        LetterboxToSquare(img_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN_LIST, std=IMAGENET_STD_LIST),
    ])


def preprocess_rgb_to_chw(img, img_size: int) -> np.ndarray:
    """
    PIL RGB → CHW float32（归一化），与 LetterboxToSquare / OpenCV letterbox 语义一致。
    """
    if getattr(img, "mode", None) is not None and img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return letterbox_rgb_u8_to_chw(arr, int(img_size))


def read_model_meta(
    pt_path: Optional[str],
    onnx_path: Optional[str],
    *,
    load_pt_meta=None,
) -> Tuple[List[str], int]:
    """
    从模型目录读取类别与输入尺寸。

    优先级：
      1. load_pt_meta 回调（PyTorch checkpoint，仅开发版）
      2. class_map.json → 类别
      3. train_config.json → img_size（及类别兜底）

    注意：存在 class_map.json 时仍必须读 train_config.json 的 img_size，
    否则 ONNX 推理会错误使用默认 224。
    """
    classes: List[str] = []
    img_size = 224

    if load_pt_meta is not None:
        classes, img_size = load_pt_meta(pt_path)

    if not classes:
        for base in filter(None, [pt_path, onnx_path]):
            map_path = Path(base).parent / "class_map.json"
            if map_path.exists():
                with open(map_path, "r", encoding="utf-8") as f:
                    classes = json.load(f).get("classes", [])
                break

    for base in filter(None, [pt_path, onnx_path]):
        cfg_path = Path(base).parent / "train_config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            img_size = cfg.get("img_size", img_size)
            if not classes:
                classes = cfg.get("classes", [])
            break

    if not classes:
        raise FileNotFoundError(
            "未能获取类别信息。请确保 class_map.json / train_config.json "
            "与模型在同一 checkpoints 目录。"
        )
    return classes, int(img_size)


def read_class_thresholds(
    pt_path: Optional[str],
    onnx_path: Optional[str],
) -> Optional[Dict[str, float]]:
    """读取 train.py 校准生成的 class_thresholds.json（可选）。"""
    for base in filter(None, [pt_path, onnx_path]):
        thr_path = Path(base).parent / "class_thresholds.json"
        if thr_path.exists():
            with open(thr_path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def build_threshold_vector(
    classes: List[str],
    class_thresholds: Optional[Dict[str, float]],
) -> Optional[np.ndarray]:
    """将逐类阈值 dict 转为与 scores 对齐的 float32 向量；无阈值时返回 None。"""
    if not class_thresholds:
        return None
    return np.array(
        [class_thresholds.get(c, 0.5) for c in classes],
        dtype=np.float32,
    )


def softmax(x: np.ndarray) -> np.ndarray:
    """数值稳定的单样本 softmax（1D logits）。"""
    x = x.astype(np.float32, copy=False)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def softmax_batch(logits: np.ndarray) -> np.ndarray:
    """批量 softmax，logits 形状 (N, C)。"""
    x = logits.astype(np.float32, copy=False)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def decide_class(
    scores: np.ndarray,
    model_classes: List[str],
    thr_vec: Optional[np.ndarray] = None,
) -> int:
    """
    在有效类别（排除 EXCLUDED_CLASSES）中取 softmax 最高分。
    thr_vec 已废弃，保留参数仅为兼容旧调用签名。
    """
    del thr_vec
    indices = active_class_indices(model_classes)
    if not indices:
        return int(np.argmax(scores))
    sub = scores[np.array(indices, dtype=np.intp)]
    return int(indices[int(np.argmax(sub))])


def build_result_dict(
    path: str,
    scores: np.ndarray,
    classes: List[str],
    thr_vec: Optional[np.ndarray],
    *,
    elapsed_ms: float = 0.0,
) -> Dict:
    """
    由 softmax 后的 scores 构造 UI / 结果管理用 dict。

    class / confidence / max_class / max_confidence 均为有效类别内 argmax；
    all_scores 仅含有效类别。
    """
    pred_idx = decide_class(scores, classes, thr_vec)
    pred_class = classes[pred_idx]
    pred_conf = float(scores[pred_idx])
    active = active_classes(classes)
    active_scores = {c: float(scores[classes.index(c)]) for c in active}
    return {
        "path": path,
        "class": pred_class,
        "class_idx": pred_idx,
        "confidence": pred_conf,
        "max_class": pred_class,
        "max_confidence": pred_conf,
        "all_scores": active_scores,
        "elapsed_ms": elapsed_ms,
        "true_class": "",
        "flagged": False,
        "correction_saved": False,
    }


def make_error_result(path: str, err: str) -> Dict:
    """推理失败时的统一结果结构（开发版 / 机台版共用）。"""
    return {
        "path": str(path),
        "class": "ERROR",
        "class_idx": -1,
        "confidence": 0.0,
        "max_class": "ERROR",
        "max_confidence": 0.0,
        "all_scores": {},
        "elapsed_ms": 0.0,
        "error": err,
        "true_class": "",
        "flagged": False,
        "correction_saved": False,
    }


def logits_row_to_result(
    path: str,
    logits_row: np.ndarray,
    classes: List[str],
    thr_vec: Optional[np.ndarray],
    *,
    elapsed_ms: float = 0.0,
) -> Dict:
    """单样本 logits → softmax → 结果 dict（ORT 输出为 raw logits，只 softmax 一次）。"""
    return build_result_dict(
        path, softmax(logits_row), classes, thr_vec, elapsed_ms=elapsed_ms,
    )


def run_batch_predict(
    image_paths: List[str],
    *,
    batch_size: int,
    preprocess_one: Callable[[str], Any],
    stack_batch: Callable[[List[Any]], Any],
    infer_batch: Callable[[Any], np.ndarray],
    classes: List[str],
    thr_vec: Optional[np.ndarray],
    progress_cb: Optional[Callable[[int, int], None]] = None,
    result_cb: Optional[Callable[[int, Dict], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> List[Dict]:
    """
    通用批量推理循环（开发 PyTorch / 开发 ONNX 回退 / 机台 ONNX 共用）。

    infer_batch 须返回 float32 logits，形状 (N, num_classes)；
    批量 softmax 后经 build_result_dict 做有效类别 argmax，避免逐行重复 softmax。
    """
    total = len(image_paths)
    results: List[Optional[Dict]] = [None] * total

    for start in range(0, total, batch_size):
        if should_stop and should_stop():
            break

        chunk_paths = image_paths[start : start + batch_size]
        tensors: List[Any] = []
        ok_indices: List[int] = []

        for offset, path in enumerate(chunk_paths):
            idx = start + offset
            try:
                tensors.append(preprocess_one(path))
                ok_indices.append(idx)
            except Exception as exc:
                err = make_error_result(str(path), str(exc))
                results[idx] = err
                if result_cb:
                    result_cb(idx, err)
                if progress_cb:
                    progress_cb(idx + 1, total)

        if not tensors:
            continue

        t0 = time.perf_counter()
        logits = infer_batch(stack_batch(tensors))
        chunk_ms = (time.perf_counter() - t0) * 1000
        per_ms = chunk_ms / len(ok_indices)
        scores = softmax_batch(np.asarray(logits))

        for j, idx in enumerate(ok_indices):
            r = build_result_dict(
                str(image_paths[idx]),
                scores[j],
                classes,
                thr_vec,
                elapsed_ms=per_ms,
            )
            results[idx] = r
            if result_cb:
                result_cb(idx, r)
            if progress_cb:
                progress_cb(idx + 1, total)

    return [
        r if r is not None else make_error_result(str(p), "未知错误")
        for r, p in zip(results, image_paths)
    ]


def run_batch_infer_chw(
    chw_list: List[Any],
    *,
    batch_size: int,
    stack_batch: Callable[[List[Any]], Any],
    infer_batch: Callable[[Any], np.ndarray],
    classes: List[str],
    thr_vec: Optional[np.ndarray],
    progress_cb: Optional[Callable[[int, int], None]] = None,
    result_cb: Optional[Callable[[int, Dict], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    paths: Optional[List[str]] = None,
) -> Tuple[List[Dict], float]:
    """
    已预处理 CHW 张量的批量推理。返回 (results, infer_wall_s)。
    infer_wall 仅含 stack + forward。
    """
    total = len(chw_list)
    if paths is None:
        paths = [f"mem:{i}" for i in range(total)]
    results: List[Optional[Dict]] = [None] * total
    infer_s = 0.0

    for start in range(0, total, batch_size):
        if should_stop and should_stop():
            break
        end = min(start + batch_size, total)
        chunk = chw_list[start:end]
        t0 = time.perf_counter()
        logits = infer_batch(stack_batch(chunk))
        chunk_s = time.perf_counter() - t0
        infer_s += chunk_s
        scores = softmax_batch(np.asarray(logits))
        per_ms = (chunk_s * 1000.0) / max(1, len(chunk))
        for j, idx in enumerate(range(start, end)):
            r = build_result_dict(
                str(paths[idx]), scores[j], classes, thr_vec, elapsed_ms=per_ms,
            )
            results[idx] = r
            if result_cb:
                result_cb(idx, r)
            if progress_cb:
                progress_cb(idx + 1, total)

    return (
        [
            r if r is not None else make_error_result(str(p), "未知错误")
            for r, p in zip(results, paths)
        ],
        infer_s,
    )


def predict_bgr_crops_timed(
    crops_bgr: List[np.ndarray],
    *,
    img_size: int,
    batch_size: int,
    stack_batch: Callable[[List[Any]], Any],
    infer_batch: Callable[[Any], np.ndarray],
    classes: List[str],
    thr_vec: Optional[np.ndarray] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    result_cb: Optional[Callable[[int, Dict], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    timing_out: Optional[Dict[str, float]] = None,
    to_model_tensor: Optional[Callable[[np.ndarray], Any]] = None,
) -> List[Dict]:
    """
    SAHI 裁剪高速路径：OpenCV letterbox（可并行）→ 批量推理。
    timing_out 写入 preprocess_s / infer_s（秒）。
    to_model_tensor: 可选，将单个 CHW numpy 转为引擎张量（如 torch.Tensor）。
    """
    chw_np, pre_s = preprocess_bgr_crops_to_chw(crops_bgr, img_size)
    if to_model_tensor is not None:
        chw_list: List[Any] = [to_model_tensor(x) for x in chw_np]
    else:
        chw_list = chw_np
    results, infer_s = run_batch_infer_chw(
        chw_list,
        batch_size=batch_size,
        stack_batch=stack_batch,
        infer_batch=infer_batch,
        classes=classes,
        thr_vec=thr_vec,
        progress_cb=progress_cb,
        result_cb=result_cb,
        should_stop=should_stop,
    )
    if timing_out is not None:
        timing_out["preprocess_s"] = float(pre_s)
        timing_out["infer_s"] = float(infer_s)
    return results
