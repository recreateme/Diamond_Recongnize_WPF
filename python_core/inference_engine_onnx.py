#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机台部署专用分类引擎（ONNX Runtime + NumPy + Pillow）。

由 app_deploy / DEFECTS_DEPLOY=1 选用；开发版见 inference_engine.py（PyTorch 优先）。
大图 SAHI 检测另走 ultralytics/torch，分类仍走本引擎（含 predict_batch_images）。
DLL 搜索统一走 app_paths.setup_ort_dll_paths()（冻结环境另有 rthook 最早兜底）。

性能要点：
  · Session 图优化 (ORT_ENABLE_ALL)
  · 批量推理：多张图一次 session.run，显著降低 GPU 启动开销
  · 预处理：与 train val 一致的 Resize(img_size)（load 时缓存尺寸）；归一化用 float32
  · 有效类别 argmax 在 load 时确定，推理时排除已废弃类别

批量循环与 logits→结果 转换见 inference_common（与开发版 ONNX 回退共用，避免逻辑漂移）。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from inference_common import (
    active_classes,
    logits_row_to_result,
    preprocess_rgb_to_chw,
    read_model_meta,
    run_batch_predict,
)

ort: Any = None
HAS_ORT = False
_ORT_IMPORT_ERROR = ""

_DEFAULT_BATCH_GPU = 64
_DEFAULT_BATCH_CPU = 8


def _cuda_hw_present() -> bool:
    """机台是否有可用 NVIDIA GPU（无独显时勿尝试 CUDA EP，避免长时间失败）。"""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _gpu_mem_limit_bytes() -> int:
    """ORT 分类显存上限：约占 25%（12GB 卡约 3GB），其余留给 YOLO/SAHI。"""
    fallback = 3 * 1024 * 1024 * 1024
    try:
        import torch

        if torch.cuda.is_available():
            total = int(torch.cuda.get_device_properties(0).total_memory)
            return max(int(total * 0.25), 1024 * 1024 * 1024)
    except Exception:
        pass
    return fallback


def _ensure_ort_import() -> bool:
    """延迟加载 onnxruntime；import 前只调用 setup_ort_dll_paths（含 preload）。"""
    global ort, HAS_ORT, _ORT_IMPORT_ERROR
    if HAS_ORT and ort is not None:
        return True
    if _ORT_IMPORT_ERROR:
        return False
    try:
        from app_paths import setup_ort_dll_paths
        setup_ort_dll_paths()
    except ImportError:
        pass
    try:
        import onnxruntime as ort_module
        ort = ort_module
        HAS_ORT = True
        return True
    except Exception as exc:
        import traceback
        _ORT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        HAS_ORT = False
        ort = None
        return False


def _ort_unavailable_message() -> str:
    if _ORT_IMPORT_ERROR:
        return (
            "onnxruntime 加载失败。\n"
            f"详情: {_ORT_IMPORT_ERROR}\n"
            "机台 GPU 包需 onnxruntime-gpu；请确认打包含 ORT DLL 或重新安装依赖。"
        )
    return "未安装 onnxruntime。机台 GPU 包需 onnxruntime-gpu。"


def ort_available_providers() -> List[str]:
    if not _ensure_ort_import():
        return []
    return ort.get_available_providers()


def pick_ort_providers(use_gpu: bool) -> Tuple[List[Any], str]:
    """选择 ORT ExecutionProvider：有独显时 CUDA 快速模式，否则 CPU。始终带 CPU 回退。"""
    available = ort_available_providers()
    if use_gpu and _cuda_hw_present() and "CUDAExecutionProvider" in available:
        cuda_opts = {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit": _gpu_mem_limit_bytes(),
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": True,
        }
        return (
            [
                ("CUDAExecutionProvider", cuda_opts),
                "CPUExecutionProvider",
            ],
            "CUDAExecutionProvider",
        )
    return (["CPUExecutionProvider"], "CPUExecutionProvider")


def _make_session_options(on_gpu: bool) -> Any:
    """创建 ORT SessionOptions：图融合提升速度；GPU 少占 CPU 线程。"""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True
    if on_gpu:
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
    else:
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
        opts.inter_op_num_threads = 1
    return opts


class InferenceEngine:
    """ONNX 推理引擎（机台版，可选 GPU，支持批量推理）。"""

    def __init__(self):
        self.ort_session = None
        self._model_classes: List[str] = []
        self.classes: List[str] = []
        self.img_size: int = 128
        self.device: str = "cpu"
        self.backend: str = "none"
        self.loaded: bool = False
        self._pt_path: Optional[str] = None
        self._onnx_path: Optional[str] = None
        self._use_gpu: bool = True
        self._ort_provider: str = "CPUExecutionProvider"
        self._input_name: str = "input"
        self._batch_size: int = _DEFAULT_BATCH_CPU

    def load(
        self,
        pt_path: Optional[str],
        onnx_path: Optional[str] = None,
        use_gpu: bool = True,
    ) -> str:
        self.loaded = False
        self._use_gpu = use_gpu

        if not _ensure_ort_import():
            raise RuntimeError(_ort_unavailable_message())
        if not onnx_path or not Path(onnx_path).exists():
            raise FileNotFoundError(
                f"未找到 ONNX 模型：{onnx_path or '(未指定)'}\n"
                "请确认 checkpoints/model.onnx（及 model.onnx.data）与 exe 同级。"
            )

        self._model_classes, self.img_size = read_model_meta(pt_path, onnx_path)
        self.classes = active_classes(self._model_classes)
        self._refresh_preprocess_cache()

        providers, picked = pick_ort_providers(use_gpu)
        on_gpu = picked == "CUDAExecutionProvider"
        sess_opts = _make_session_options(on_gpu)
        session_error = ""
        try:
            self.ort_session = ort.InferenceSession(
                str(onnx_path),
                sess_options=sess_opts,
                providers=providers,
            )
        except Exception as exc:
            if not use_gpu:
                raise
            session_error = f"{type(exc).__name__}: {exc}"
            self.ort_session = ort.InferenceSession(
                str(onnx_path),
                sess_options=_make_session_options(False),
                providers=["CPUExecutionProvider"],
            )
        inputs = self.ort_session.get_inputs()
        self._input_name = inputs[0].name if inputs else "input"

        active = self.ort_session.get_providers()
        self._ort_provider = active[0] if active else picked
        self.device = "cuda" if "CUDA" in self._ort_provider.upper() else "cpu"
        self._batch_size = (
            _DEFAULT_BATCH_GPU if self.device == "cuda" else _DEFAULT_BATCH_CPU
        )
        self.backend = "onnx"
        self._pt_path = pt_path
        self._onnx_path = onnx_path
        self.loaded = True

        dev_label = "GPU" if self.device == "cuda" else "CPU"
        extra = ""
        if self.device == "cuda":
            try:
                import torch

                extra = f" · {torch.cuda.get_device_name(0)}"
            except Exception:
                extra = ""
        if use_gpu and self.device != "cuda":
            if session_error:
                hint = (
                    f"模型加载成功  [ONNX / {dev_label}]  {len(self.classes)} 类别 "
                    f"· {self.img_size}px · batch={self._batch_size}\n"
                    f"（无可用 GPU 或 CUDA 初始化失败，已回退 CPU：{session_error}）"
                )
            else:
                hint = (
                    f"模型加载成功  [ONNX / {dev_label}]  {len(self.classes)} 类别 "
                    f"· {self.img_size}px · batch={self._batch_size}\n"
                    f"（未检测到独立显卡，已使用 CPU。有 NVIDIA GPU 时将自动走 CUDA 快速模式）"
                )
        else:
            hint = (
                f"模型加载成功  [ONNX / {dev_label}{extra}]  {len(self.classes)} 类别 "
                f"· {self.img_size}px · batch={self._batch_size}"
            )
        return hint

    def reload(self) -> str:
        return self.load(self._pt_path, self._onnx_path, use_gpu=self._use_gpu)

    def _refresh_preprocess_cache(self) -> None:
        # 尺寸已由 self.img_size 驱动；保留钩子供 load 调用，避免调用点分叉
        pass

    def predict(self, image_path: str) -> Dict:
        if not self.loaded:
            raise RuntimeError("模型未加载，请先在「设置」页面加载模型。")
        t0 = time.perf_counter()
        logits = self.ort_session.run(
            None, {self._input_name: self._preprocess(image_path)},
        )[0][0]
        result = logits_row_to_result(
            str(image_path), logits, self._model_classes, None,
        )
        result["elapsed_ms"] = (time.perf_counter() - t0) * 1000
        return result

    def predict_batch(
        self,
        image_paths: List[str],
        progress_cb: Optional[Callable[[int, int], None]] = None,
        result_cb: Optional[Callable[[int, Dict], None]] = None,
        batch_size: Optional[int] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> List[Dict]:
        if not self.loaded:
            raise RuntimeError("模型未加载，请先在「设置」页面加载模型。")

        session = self.ort_session
        input_name = self._input_name

        def _infer_batch(batch: np.ndarray) -> np.ndarray:
            return session.run(None, {input_name: batch})[0]

        return run_batch_predict(
            image_paths,
            batch_size=batch_size or self._batch_size,
            preprocess_one=self._preprocess_chw,
            stack_batch=lambda ts: np.stack(ts, axis=0).astype(np.float32, copy=False),
            infer_batch=_infer_batch,
            classes=self._model_classes,
            thr_vec=None,
            progress_cb=progress_cb,
            result_cb=result_cb,
            should_stop=should_stop,
        )

    def predict_batch_images(
        self,
        images: List[Image.Image],
        progress_cb: Optional[Callable[[int, int], None]] = None,
        result_cb: Optional[Callable[[int, Dict], None]] = None,
        batch_size: Optional[int] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> List[Dict]:
        """
        内存图像批量推理（SAHI 裁剪用）。
        与 predict_batch / 开发版引擎同一套 classes、阈值与 preprocess。
        """
        if not self.loaded:
            raise RuntimeError("模型未加载，请先在「设置」页面加载模型。")
        rgb = [im.convert("RGB") if im.mode != "RGB" else im for im in images]
        keys = [f"mem:{i}" for i in range(len(rgb))]

        def _preprocess_chw(key: str) -> np.ndarray:
            idx = int(key.split(":", 1)[1])
            return self._preprocess_image(rgb[idx])

        session = self.ort_session
        input_name = self._input_name

        def _infer_batch(batch: np.ndarray) -> np.ndarray:
            return session.run(None, {input_name: batch})[0]

        return run_batch_predict(
            keys,
            batch_size=batch_size or self._batch_size,
            preprocess_one=_preprocess_chw,
            stack_batch=lambda ts: np.stack(ts, axis=0).astype(np.float32, copy=False),
            infer_batch=_infer_batch,
            classes=self._model_classes,
            thr_vec=None,
            progress_cb=progress_cb,
            result_cb=result_cb,
            should_stop=should_stop,
        )

    def _preprocess(self, image_path: str) -> np.ndarray:
        return self._preprocess_chw(image_path)[np.newaxis]

    def _preprocess_chw(self, image_path: str) -> np.ndarray:
        return self._preprocess_image(self._load_rgb(image_path))

    def _load_rgb(self, image_path: str) -> Image.Image:
        with Image.open(image_path) as im:
            return im.convert("RGB")

    def _preprocess_image(self, img: Image.Image) -> np.ndarray:
        return preprocess_rgb_to_chw(img, self.img_size)
