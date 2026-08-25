#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从输出目录下一层子文件夹中的 statistics.json 重建 summary.csv。

用法:
  · 将本程序（或打包后的 .exe）放在结果根目录（如 D:\\迅雷下载\\ECOA）后双击
  · 或: python rebuild_summary_from_stats.py [结果根目录]

扫描规则: 仅 <根目录>/*/statistics.json（一层子目录）
输出: <根目录>/summary.csv（若已存在则先备份为 summary.csv.bak）
CSV 格式与当前应用钻石检测分类模块一致。
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CLASS_COLUMNS = ("棱边朝上", "点朝上", "面朝上")
HEADER = (
    "图像",
    "汇总钻石数",
    *CLASS_COLUMNS,
    "检测耗时(s)",
    "分类耗时(s)",
    "总耗时(s)",
)


def base_dir(argv: List[str]) -> Path:
    if len(argv) > 1 and argv[1].strip():
        return Path(argv[1]).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd().resolve()


def natural_key(text: str) -> Tuple:
    parts = re.split(r"(\d+)", text)
    key: List[Any] = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return tuple(key)


def load_stats(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"  [跳过] 非对象 JSON: {path}")
            return None
        return data
    except Exception as exc:
        print(f"  [跳过] 读取失败 {path}: {exc}")
        return None


def row_from_stats(data: Dict[str, Any], fallback_name: str) -> Dict[str, Any]:
    counts = data.get("defect_counts") or {}
    if not isinstance(counts, dict):
        counts = {}
    image = str(data.get("image") or fallback_name)
    return {
        "image": image,
        "total": int(data.get("total_diamonds") or 0),
        "classes": {c: int(counts.get(c) or 0) for c in CLASS_COLUMNS},
        "det": float(data.get("detection_time_s") or 0),
        "cls": float(data.get("classification_time_s") or 0),
        "total_t": float(data.get("total_time_s") or 0),
    }


def fmt_time(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".") if v else "0"


def write_summary_csv(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    if out_path.exists():
        bak = out_path.with_name(out_path.name + ".bak")
        shutil.copy2(out_path, bak)
        print(f"已备份原文件 -> {bak}")

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        class_totals = {c: 0 for c in CLASS_COLUMNS}
        diamond_total = 0
        for r in rows:
            diamond_total += r["total"]
            for c in CLASS_COLUMNS:
                class_totals[c] += r["classes"][c]
            writer.writerow(
                [
                    r["image"],
                    r["total"],
                    *(r["classes"][c] for c in CLASS_COLUMNS),
                    fmt_time(r["det"]),
                    fmt_time(r["cls"]),
                    fmt_time(r["total_t"]),
                ]
            )
        if len(rows) > 1:
            writer.writerow(
                [
                    "批次合计",
                    diamond_total,
                    *(class_totals[c] for c in CLASS_COLUMNS),
                    "",
                    "",
                    "",
                ]
            )


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    root = base_dir(argv)
    print(f"结果根目录: {root}")
    if not root.is_dir():
        print(f"错误: 目录不存在: {root}")
        return 1

    candidates = sorted(
        root.glob("*/statistics.json"),
        key=lambda p: natural_key(p.parent.name),
    )
    print(f"找到 {len(candidates)} 个 statistics.json（一层子目录）")
    if not candidates:
        print("未找到 */statistics.json，未写入 CSV。")
        return 1

    rows: List[Dict[str, Any]] = []
    for path in candidates:
        print(f"  · {path.relative_to(root)}")
        data = load_stats(path)
        if data is None:
            continue
        rows.append(row_from_stats(data, fallback_name=f"{path.parent.name}.jpg"))

    if not rows:
        print("没有成功解析的统计文件，未写入 CSV。")
        return 1

    out_csv = root / "summary.csv"
    write_summary_csv(out_csv, rows)
    print(f"已写入: {out_csv}")
    print(f"共 {len(rows)} 张图" + ("（含批次合计行）" if len(rows) > 1 else "（单张，无批次合计）"))
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception as exc:
        print(f"错误: {exc}")
        code = 1
    finally:
        try:
            input("\n按回车键退出…")
        except EOFError:
            pass
    raise SystemExit(code)
