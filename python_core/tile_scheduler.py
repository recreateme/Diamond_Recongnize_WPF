"""
整盘批量调度：只处理"可能有钻石"的 tile，跳过盘外空白图
====================================================

背景：一个盘拍成 24×24 = 576 张 tile，圆盘内接于网格，四角约四分之一是盘外空白。
实测一盘（51355 颗）里 154 张是 0 颗，检测耗时与有无钻石几乎无关（约 3.4 s/张），
这些图每盘白白花掉约 9.5 分钟（占整批 15%）。

做法（邻域扩展 + 抽检回查，不依赖任何图像阈值，不需要标定）：

  1. 从网格中心附近的 tile 开始处理；
  2. 只有"检出钻石"的 tile，才把它的 8 邻域里还没处理的 tile 排进队列；
  3. 队列空了就结束——没被排进来的 tile 离任何含钻 tile 都至少隔着一整张图，
     而圆盘是凸的，不可能在中间"断开"，所以它们是盘外空白；
  4. 抽检回查：结束后随机挑几张被跳过的 tile（优先离盘最近的）真正处理一次。
     只要有一张抽到钻石，就认为"盘不凸/编号规则不符/批次不是单盘"，
     立刻退回"处理全部剩余 tile"，宁可慢也不漏。

为什么不会漏：圆盘覆盖的 tile 集合是 8 连通的（凸区域），每个含钻 tile 都能从
中心沿含钻 tile 一路扩展到。邻接关系只依赖"网格上谁挨着谁"，所以行优先/列优先
编号结果完全一样（互为转置）；**但蛇形编号会让邻接关系错位**，默认编号按行优先
（已用整盘 summary.csv 验证：行优先映射出的占用图是光滑的圆，蛇形则锯齿）。

自动关闭（退回逐张处理）的情形：文件名取不出 tile 编号、编号重复或越界、
批次覆盖不到网格的 min_coverage（比如只选了几十张）。

接口（逐张循环，兼容 pythonnet 动态调用）：
    sch = TileScheduler(names, rows=24, cols=24)
    while True:
        i = sch.next_index()          # 原 names 里的下标；-1 表示结束
        if i < 0: break
        n = process(names[i])         # 返回检出钻石数
        sch.report(i, n, error=False)
    sch.skipped_indices()             # 没被处理的下标（盘外空白图）
"""
from __future__ import annotations

import os
import random
import re
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Sequence, Set, Tuple

_SCHEMES = ("row_major", "row_snake", "col_major", "col_snake")


def _index_to_rc(n: int, rows: int, cols: int, scheme: str) -> Tuple[int, int]:
    k = n - 1
    if scheme == "row_major":
        r, c = divmod(k, cols)
    elif scheme == "row_snake":
        r, c = divmod(k, cols)
        if r % 2 == 1:
            c = cols - 1 - c
    elif scheme == "col_major":
        c, r = divmod(k, rows)
    elif scheme == "col_snake":
        c, r = divmod(k, rows)
        if c % 2 == 1:
            r = rows - 1 - r
    else:
        raise ValueError(f"未知编号规则 {scheme}")
    return r, c


_NUM_RE = re.compile(r"(\d+)$")


class TileScheduler:
    def __init__(
        self,
        names: Sequence[str],
        rows: int = 24,
        cols: int = 24,
        scheme: str = "row_major",
        audit_count: int = 3,
        min_coverage: float = 0.8,
        seed: Optional[int] = 0,
        prescreen: Optional[Callable[[int], bool]] = None,
    ):
        """
        names：本批所有图像文件名（顺序即原输入顺序）。
        audit_count：抽检回查张数（0 = 不抽检）。
        prescreen(idx)->bool：可选的廉价预判，返回 True 表示"肯定是空的，别跑检测"。
        """
        self.names: List[str] = [str(n) for n in names]
        self.rows, self.cols = int(rows), int(cols)
        self.scheme = scheme
        self.audit_count = max(0, int(audit_count))
        self._rng = random.Random(seed)
        self._prescreen = prescreen

        self.enabled = False
        self.reason = ""
        self._rc: Dict[int, Tuple[int, int]] = {}      # 下标 -> (r,c)
        self._at: Dict[Tuple[int, int], int] = {}      # (r,c) -> 下标
        self._always: List[int] = []                   # 取不出编号的图：一律处理
        self._visited: Set[int] = set()
        self._nonempty: Set[int] = set()
        self._queue: Deque[int] = deque()
        self._queued: Set[int] = set()
        self._phase = "expand"
        self._audit: List[int] = []
        self._audited: List[int] = []
        self.audit_failed = False
        self._prescreen_skipped: List[int] = []
        self._seq = 0                                   # 禁用时的顺序游标

        self._build_map(min_coverage)

    # ---------------------------------------------------------------- 初始化

    def _build_map(self, min_coverage: float) -> None:
        total_cells = self.rows * self.cols
        seen: Dict[int, int] = {}
        for i, name in enumerate(self.names):
            stem = os.path.splitext(os.path.basename(name))[0]
            m = _NUM_RE.search(stem)
            if not m:
                self._always.append(i)
                continue
            num = int(m.group(1))
            if num < 1 or num > total_cells or num in seen:
                self.reason = f"tile 编号越界或重复（{name}）"
                return
            seen[num] = i
        for num, i in seen.items():
            rc = _index_to_rc(num, self.rows, self.cols, self.scheme)
            self._rc[i] = rc
            self._at[rc] = i
        if len(self._rc) < min_coverage * total_cells:
            self.reason = f"本批只覆盖 {len(self._rc)}/{total_cells} 个 tile，不是整盘，逐张处理"
            return
        self.enabled = True

    # ---------------------------------------------------------------- 迭代

    def _center_dist(self, i: int) -> float:
        r, c = self._rc[i]
        return (r - (self.rows - 1) / 2) ** 2 + (c - (self.cols - 1) / 2) ** 2

    def _neighbors(self, i: int) -> List[int]:
        r, c = self._rc[i]
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                j = self._at.get((r + dr, c + dc))
                if j is not None:
                    out.append(j)
        return out

    def _take(self, i: int) -> int:
        self._visited.add(i)
        return i

    def next_index(self) -> int:
        """下一张要处理的图的下标；-1 表示全部结束。"""
        if not self.enabled:
            if self._seq >= len(self.names):
                return -1
            i = self._seq
            self._seq += 1
            return i

        while True:
            if self._phase == "expand":
                i = self._next_expand()
                if i is not None:
                    if self._prescreen is not None and self._prescreen(i):
                        self._visited.add(i)
                        self._prescreen_skipped.append(i)
                        continue
                    return self._take(i)
                self._begin_audit()
            if self._phase == "audit":
                while self._audit:
                    i = self._audit.pop()
                    if i not in self._visited:
                        self._audited.append(i)
                        return self._take(i)
                self._phase = "always"
            if self._phase == "rest":
                rest = [i for i in sorted(self._rc) if i not in self._visited]
                if rest:
                    return self._take(rest[0])
                self._phase = "always"
            if self._phase == "always":
                while self._always:
                    i = self._always.pop(0)
                    if i not in self._visited:
                        return self._take(i)
                self._phase = "done"
            if self._phase == "done":
                return -1

    def _next_expand(self) -> Optional[int]:
        while self._queue:
            i = self._queue.popleft()
            if i not in self._visited:
                return i
        if not self._nonempty:
            # 还没发现任何含钻 tile（起点恰好是空的）：按离中心由近到远逐张找，
            # 找到第一张含钻的就转入邻域扩展。最坏情形 = 逐张处理，不比原来慢。
            cand = [i for i in self._rc if i not in self._visited]
            if cand:
                return min(cand, key=lambda i: (self._center_dist(i), i))
        return None

    def _begin_audit(self) -> None:
        unvisited = [i for i in self._rc if i not in self._visited]
        if not unvisited or self.audit_count == 0 or not self._nonempty:
            self._phase = "always"
            return
        # 优先抽离含钻 tile 切比雪夫距离 ≤2 的（最容易出问题的盘边一圈）
        occ = [self._rc[i] for i in self._nonempty]
        near = []
        for i in unvisited:
            r, c = self._rc[i]
            if any(max(abs(r - orow), abs(c - ocol)) <= 2 for orow, ocol in occ):
                near.append(i)
        pool = near if len(near) >= self.audit_count else unvisited
        k = min(self.audit_count, len(pool))
        self._audit = self._rng.sample(pool, k)
        self._phase = "audit"

    # ---------------------------------------------------------------- 回报

    def report(self, i: int, n_diamonds: int, error: bool = False) -> None:
        """回报第 i 张的检出数。检测出错按"可能有钻"处理（保守地继续扩展）。"""
        if not self.enabled or i not in self._rc:
            return
        has = error or n_diamonds > 0
        if has:
            self._nonempty.add(i)
        if self._phase == "audit" and i in self._audited and has:
            # 抽检抽到了钻石：邻域扩展的前提不成立，退回处理全部剩余 tile
            self.audit_failed = True
            self._phase = "rest"
            self._audit = []
            return
        if self._phase == "expand" and has:
            for j in self._neighbors(i):
                if j not in self._visited and j not in self._queued:
                    self._queue.append(j)
                    self._queued.add(j)

    # ---------------------------------------------------------------- 结果

    def skipped_indices(self) -> List[int]:
        """没被处理的下标（盘外空白图），按下标升序。"""
        if not self.enabled:
            return []
        return sorted(i for i in self._rc if i not in self._visited or i in self._prescreen_skipped)

    def status_text(self) -> str:
        if self.enabled:
            return (f"整盘批量：{len(self._rc)} 张，按邻域扩展跳过盘外空白图"
                    f"（抽检 {self.audit_count} 张）")
        return f"逐张处理（{self.reason or '未启用跳过'}）"

    def summary_text(self) -> str:
        if not self.enabled:
            return self.status_text()
        sk = len(self.skipped_indices())
        s = f"跳过 {sk} 张盘外空白图，抽检 {len(self._audited)} 张"
        if self.audit_failed:
            s += "；抽检发现含钻图，已退回处理全部剩余图像（请检查编号规则或批次是否为单盘）"
        return s

    def summary(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "reason": self.reason,
            "n_total": len(self.names),
            "n_skipped": len(self.skipped_indices()),
            "n_audited": len(self._audited),
            "audit_failed": self.audit_failed,
        }


def make_thumb_prescreen(
    paths: Sequence[str], threshold: float, reduce: int = 8
) -> Callable[[int], bool]:
    """可选的廉价预判：读 1/8 缩略图算内容分数，低于 threshold 判空。

    只用于进一步省掉"紧贴盘边的空白图"（邻域扩展必须处理的那一圈）。
    阈值必须用真实图标定（disk_stitch.calibrate_threshold），并保证漏判率为 0。
    默认不启用。
    """
    from disk_stitch import thumb_content_score  # 延迟导入，避免硬依赖 OpenCV

    def _fn(i: int) -> bool:
        try:
            return thumb_content_score(paths[i], reduce=reduce) < threshold
        except Exception:
            return False  # 读不了就老实检测

    return _fn
