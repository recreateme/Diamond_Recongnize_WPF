#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化测试：读取产品原图 + boxes_opencv.json 中第一个检测框，画框并展示。

默认路径（可改命令行参数）:
  图像: D:\\001产品完整数据\\12.jpg
  框:   D:\\迅雷下载\\ECOA\\12\\boxes_opencv.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(description="测试：在原图上绘制第一个 OpenCV 框并展示")
    parser.add_argument(
        "--image",
        default=r"D:\001产品完整数据\12.jpg",
        help="原图路径",
    )
    parser.add_argument(
        "--boxes",
        default=r"D:\迅雷下载\ECOA\12\boxes_opencv.json",
        help="boxes_opencv.json 路径",
    )
    parser.add_argument(
        "--out",
        default="",
        help="可选：保存可视化结果路径；默认与 boxes 同目录 vis_first_box.jpg",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="只保存不弹窗",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    boxes_path = Path(args.boxes)
    if not image_path.is_file():
        print(f"找不到图像: {image_path}")
        return 1
    if not boxes_path.is_file():
        print(f"找不到框文件: {boxes_path}")
        return 1

    with open(boxes_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    boxes = data.get("boxes") or []
    if not boxes:
        print("boxes 列表为空")
        return 1

    box = boxes[0]
    x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
    print(f"图像: {image_path}")
    print(f"第一个目标 id={box.get('id')}  (x1,y1,x2,y2)=({x1},{y1},{x2},{y2})")

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"OpenCV 无法读取图像: {image_path}")
        return 1

    h, w = img.shape[:2]
    line_w = max(2, max(h, w) // 640)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), line_w)
    label = f"id={box.get('id')} ({x1},{y1})-({x2},{y2})"
    font_scale = max(0.6, max(h, w) / 2000.0)
    cv2.putText(
        img, label, (x1, max(y1 - 10, 30)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), max(1, line_w // 2),
        cv2.LINE_AA,
    )

    out_path = Path(args.out) if args.out else boxes_path.parent / "vis_first_box.jpg"
    cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"已保存: {out_path}")

    if not args.no_show:
        # 大图缩放到可显示窗口
        max_side = 1280
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            disp = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            disp = img
        win = "first box (press any key)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, disp)
        print("窗口已打开，按任意键关闭…")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
