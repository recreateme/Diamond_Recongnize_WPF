#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MobileNetV3-Small 分类骨干 — 训练与开发版 PyTorch 推理共用。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRETRAINED = REPO_ROOT / "checkpoints" / "pretrained" / "mobilenet_v3_small.pth"
MODEL_ARCH = "mobilenet_v3_small"


def build_mobilenet_v3_small(
    num_classes: int,
    *,
    pretrained: bool = True,
    weights_path: Optional[Path] = None,
) -> nn.Module:
    """
    构建 MobileNetV3-Small，替换末层 Linear 为 num_classes。

    pretrained=True 时从本地 ImageNet 权重加载（不访问网络）。
    """
    model = models.mobilenet_v3_small(weights=None)
    if pretrained:
        path = Path(weights_path) if weights_path else DEFAULT_PRETRAINED
        if not path.is_file():
            raise FileNotFoundError(
                f"未找到 MobileNetV3-Small 预训练权重: {path}\n"
                f"请将 torchvision 权重复制为:\n  {DEFAULT_PRETRAINED}"
            )
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    print(f"  模型   : MobileNetV3-Small  (in_features={in_features}, classes={num_classes})")
    return model
