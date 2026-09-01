#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从钻石检测输出的 result.json 提取 OpenCV 绝对框坐标并另存。

坐标系与应用内一致：原图左上为原点，x 向右、y 向下；导出为 int，可直接用于
cv2.rectangle(img, (x1, y1), (x2, y2), ...)。

用法:
  python export_boxes_from_result.py <result.json>
  python export_boxes_from_result.py <批次根目录>   # 扫描一层 */result.json
  python export_boxes_from_result.py               # 默认当前目录（单文件或批次根）

每个 result.json 同级写出:
  boxes_opencv.json  — 仅坐标列表
  boxes_opencv.csv   — 同上，CSV 便于查看
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def to_int_box(det: Dict[str, Any]) -> Tuple[int, int, int, int]:
    """与 sahi_detector.draw_* 一致：int() 截断。"""
    return (
        int(det["x1"]),
        int(det["y1"]),
        int(det["x2"]),
        int(det["y2"]),
    )


def extract_boxes(result: Dict[str, Any]) -> List[Dict[str, int]]:
    boxes: List[Dict[str, int]] = []
    for i, det in enumerate(result.get("detections") or [], start=1):
        if not isinstance(det, dict):
            continue
        if not all(k in det for k in ("x1", "y1", "x2", "y2")):
            continue
        x1, y1, x2, y2 = to_int_box(det)
        boxes.append({"id": i, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return boxes


def write_outputs(out_dir: Path, image_name: str, boxes: List[Dict[str, int]]) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "boxes_opencv.json"
    csv_path = out_dir / "boxes_opencv.csv"

    payload = {
        "image": image_name,
        "count": len(boxes),
        "coordinate_system": "opencv_abs_xyxy",
        "note": "x1,y1,x2,y2 are int(); origin top-left; same as cv2.rectangle",
        "boxes": boxes,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x1", "y1", "x2", "y2"])
        for b in boxes:
            writer.writerow([b["id"], b["x1"], b["y1"], b["x2"], b["y2"]])

    return json_path, csv_path


def process_result_json(path: Path) -> int:
    print(f"读取: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"  [失败] 无法解析 JSON: {exc}")
        return 0

    if not isinstance(data, dict):
        print("  [失败] result.json 根节点不是对象")
        return 0

    boxes = extract_boxes(data)
    image_name = str(data.get("image") or path.parent.name)
    json_path, csv_path = write_outputs(path.parent, image_name, boxes)
    print(f"  框数: {len(boxes)}")
    print(f"  -> {json_path}")
    print(f"  -> {csv_path}")
    return 1


def collect_targets(target: Path) -> List[Path]:
    if target.is_file():
        if target.name.lower() != "result.json":
            raise ValueError(f"文件名应为 result.json，收到: {target.name}")
        return [target.resolve()]

    if not target.is_dir():
        raise FileNotFoundError(f"路径不存在: {target}")

    # 目录本身若有 result.json，也处理
    direct = target / "result.json"
    found: List[Path] = []
    if direct.is_file():
        found.append(direct.resolve())

    # 一层子目录 */result.json（批次根）
    for child in sorted(target.iterdir()):
        if not child.is_dir():
            continue
        cand = child / "result.json"
        if cand.is_file():
            found.append(cand.resolve())

    # 去重并保持顺序
    seen = set()
    unique: List[Path] = []
    for p in found:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="从 result.json 导出 OpenCV 绝对框坐标 (boxes_opencv.json / .csv)",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="result.json 路径，或含 */result.json 的批次根目录（默认当前目录）",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    target = Path(args.path).expanduser().resolve()
    try:
        targets = collect_targets(target)
    except (FileNotFoundError, ValueError) as exc:
        print(f"错误: {exc}")
        return 1

    if not targets:
        print(f"未找到 result.json: {target}")
        print("提示: 传入单个 result.json，或传入含一层子目录的批次根（如 defects）。")
        return 1

    print(f"共 {len(targets)} 个 result.json")
    ok = 0
    for p in targets:
        ok += process_result_json(p)

    print(f"完成: 成功 {ok}/{len(targets)}")
    return 0 if ok == len(targets) else 1


if __name__ == "__main__":
    raise SystemExit(main())
