#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EfficientNetV2-S 分类骨干 — 训练与开发版 PyTorch 推理共用。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRETRAINED = (
    REPO_ROOT / "checkpoints" / "pretrained" / "efficientnet_v2_s-dd5fe13b.pth"
)
MODEL_ARCH = "efficientnet_v2_s"


def build_efficientnet_v2_s(
    num_classes: int,
    *,
    pretrained: bool = True,
    weights_path: Optional[Path] = None,
) -> nn.Module:
    """
    构建 EfficientNetV2-S，替换末层 Linear 为 num_classes。

    pretrained=True 时从本地 ImageNet 权重加载（不访问网络）。
    """
    model = models.efficientnet_v2_s(weights=None)
    if pretrained:
        path = Path(weights_path) if weights_path else DEFAULT_PRETRAINED
        if not path.is_file():
            raise FileNotFoundError(
                f"未找到 EfficientNetV2-S 预训练权重: {path}\n"
                f"请下载:\n"
                f"  https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth\n"
                f"并保存为:\n  {DEFAULT_PRETRAINED}"
            )
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    print(
        f"  模型   : EfficientNetV2-S  "
        f"(in_features={in_features}, classes={num_classes})"
    )
    return model


def build_model(
    num_classes: int,
    *,
    pretrained: bool = True,
    weights_path: Optional[Path] = None,
) -> nn.Module:
    """统一入口：当前默认骨干 EfficientNetV2-S。"""
    return build_efficientnet_v2_s(
        num_classes, pretrained=pretrained, weights_path=weights_path
    )
