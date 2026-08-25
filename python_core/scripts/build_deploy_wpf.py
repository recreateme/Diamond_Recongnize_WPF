#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WPF 机台包组装入口（调用 scripts/build_deploy_wpf.ps1）。

用法（在仓库根或 python_core 下）:
  python scripts/build_deploy_wpf.py
  python scripts/build_deploy_wpf.py --skip-python-runtime
  python scripts/build_deploy_wpf.py --python-home D:\\Software\\MiniAnaconda\\envs\\cv-yolo
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 本文件位于 python_core/scripts/ → 仓库根为 parents[2]
# 若复制到仓库 scripts/ 则 parents[1]
if (HERE.parent / "inference_common.py").exists():
    ROOT = HERE.parent.parent
    PS1 = ROOT / "scripts" / "build_deploy_wpf.ps1"
else:
    ROOT = HERE.parent
    PS1 = HERE / "build_deploy_wpf.ps1"


def main() -> int:
    ap = argparse.ArgumentParser(description="Assemble WPF machine package")
    ap.add_argument("--python-home", default="", help="Conda env root to copy as python_runtime")
    ap.add_argument("--skip-publish", action="store_true")
    ap.add_argument("--skip-python-runtime", action="store_true")
    ap.add_argument("--yolo-path", default="")
    ap.add_argument("--configuration", default="Release", choices=("Debug", "Release"))
    args = ap.parse_args()

    if not PS1.is_file():
        print(f"找不到 {PS1}", file=sys.stderr)
        return 1

    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(PS1),
        "-Configuration",
        args.configuration,
    ]
    if args.python_home:
        cmd += ["-PythonHome", args.python_home]
    if args.skip_publish:
        cmd.append("-SkipPublish")
    if args.skip_python_runtime:
        cmd.append("-SkipPythonRuntime")
    if args.yolo_path:
        cmd += ["-YoloPath", args.yolo_path]

    print(">", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
