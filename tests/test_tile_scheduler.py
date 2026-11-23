"""
tile_scheduler 验证：回放一盘真实占用图 + 随机盘 + 异常情形。

运行：python tests/test_tile_scheduler.py   （或 pytest）

REAL_MAP 来自一盘真实批量结果的 summary.csv（51355 颗，576 张，其中 154 张为 0 颗），
'#' = 该 tile 检出钻石，'.' = 0 颗。行优先编号：第 r 行第 c 列 = 编号 r*24+c+1。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python_core"))
from tile_scheduler import TileScheduler  # noqa: E402

REAL_MAP = """
........######..........
.....############.......
.....##############.....
....################....
...##################...
..###################...
.#####################..
.#####################..
.######################.
#######################.
#######################.
#######################.
#######################.
#######################.
#######################.
.######################.
.#####################..
..####################..
..###################...
...#################....
....###############.....
.....#############......
........########........
........................
""".strip().splitlines()

R = C = 24


def real_counts():
    cnt = {}
    for r, line in enumerate(REAL_MAP):
        for c, ch in enumerate(line):
            cnt[r * C + c + 1] = 100 if ch == "#" else 0
    return cnt


def names_for(cnt):
    # 故意按字典序给，和实际批量（1,10,100,101…）一致
    return sorted((f"{k}.jpg" for k in cnt), key=str)


def _num(name):
    stem = name.rsplit(".", 1)[0]
    return int(stem) if stem.isdigit() else -1


def run(names, cnt_by_num, **kw):
    sch = TileScheduler(names, R, C, **kw)
    processed = []
    guard = 0
    while True:
        i = sch.next_index()
        if i < 0:
            break
        processed.append(i)
        sch.report(i, cnt_by_num.get(_num(names[i]), 0))
        guard += 1
        assert guard <= len(names) + 5, "调度器没有收敛"
    return sch, processed


def recall(names, cnt_by_num, processed):
    ps = set(processed)
    need = {i for i, n in enumerate(names) if cnt_by_num.get(int(n.split(".")[0]), 0) > 0}
    return len(need - ps), len(need)


def test_real_disc():
    cnt = real_counts()
    names = names_for(cnt)
    sch, proc = run(names, cnt)
    missed, need = recall(names, cnt, proc)
    assert sch.enabled
    assert missed == 0, f"漏掉 {missed}/{need} 张含钻图"
    assert len(proc) + len(sch.skipped_indices()) == len(names)
    # 没有抽检时跳过 74 张（空白且 8 邻域都空）；3 张抽检要处理，所以最多 74-3
    sch0, proc0 = run(names, cnt, audit_count=0)
    assert len(sch0.skipped_indices()) == 74, len(sch0.skipped_indices())
    assert recall(names, cnt, proc0)[0] == 0
    print(f"  真实占用图：处理 {len(proc)} / 576，跳过 {len(sch.skipped_indices())}，"
          f"不抽检时跳过 {len(sch0.skipped_indices())}，漏检 0/{need}")


def disc_counts(cr, cc, rad):
    cnt = {}
    for r in range(R):
        for c in range(C):
            # tile 与圆相交就算含钻（取 tile 内离圆心最近点）
            ny = min(max(cr, r), r + 1)
            nx = min(max(cc, c), c + 1)
            inside = (ny - cr) ** 2 + (nx - cc) ** 2 <= rad * rad
            cnt[r * C + c + 1] = 50 if inside else 0
    return cnt


def test_random_discs_and_shifts():
    rng = random.Random(7)
    worst_saved, bad = 1.0, 0
    for _ in range(300):
        rad = rng.uniform(3.0, 12.0)
        cr = rng.uniform(rad * 0.3, R - rad * 0.3)   # 允许圆被网格边界截掉一部分
        cc = rng.uniform(rad * 0.3, C - rad * 0.3)
        cnt = disc_counts(cr, cc, rad)
        if sum(1 for v in cnt.values() if v) == 0:
            continue
        names = names_for(cnt)
        sch, proc = run(names, cnt, audit_count=0)
        m, need = recall(names, cnt, proc)
        if m:
            bad += 1
        worst_saved = min(worst_saved, 1 - len(proc) / len(names))
    assert bad == 0, f"{bad} 个随机盘出现漏检"
    print("  300 个随机位置/大小的盘（含偏心、被网格截断）：漏检 0")


def test_center_empty_falls_back():
    # 盘在左上角，网格中心是空的
    cnt = disc_counts(4.0, 4.0, 3.5)
    assert cnt[12 * C + 12 + 1] == 0
    names = names_for(cnt)
    sch, proc = run(names, cnt, audit_count=0)
    assert recall(names, cnt, proc)[0] == 0
    print(f"  盘在角落（中心 tile 为空）：处理 {len(proc)}，漏检 0")


def test_audit_catches_island():
    cnt = real_counts()
    # 在盘外造一个孤岛：右下角 (22,22) 有钻，离大盘距离 ≥2，邻域扩展会漏掉它
    cnt[22 * C + 22 + 1] = 40
    names = names_for(cnt)
    sch_off, proc_off = run(names, cnt, audit_count=0)
    assert recall(names, cnt, proc_off)[0] == 1, "孤岛应被邻域扩展漏掉（这正是需要抽检的原因）"
    sch, proc = run(names, cnt, audit_count=400, seed=1)  # 抽检张数足够大，必然抽到
    assert sch.audit_failed
    assert recall(names, cnt, proc)[0] == 0, "抽检失败后应退回处理全部剩余图像"
    print("  孤岛场景：不抽检会漏 1 张；抽检触发后退回全处理，漏检 0")


def test_disabled_cases():
    cnt = real_counts()
    names = names_for(cnt)
    # 只选了 100 张 → 不是整盘
    sch, proc = run(names[:100], cnt)
    assert not sch.enabled and len(proc) == 100 and not sch.skipped_indices()
    # 文件名没有编号
    sch2, proc2 = run([f"IMG_{k}_a.jpg" for k in range(600)], {})
    assert not sch2.enabled and len(proc2) == 600
    # 编号越界
    sch3, proc3 = run([f"{k}.jpg" for k in range(1, 700)], {})
    assert not sch3.enabled
    # 混入取不出编号的图：必须被处理
    names4 = names + ["stray.jpg"]
    sch4, proc4 = run(names4, cnt)
    assert names4.index("stray.jpg") in proc4
    print("  自动关闭：非整盘 / 无编号 / 越界 / 混入杂图 均正确")


def test_error_is_conservative():
    cnt = real_counts()
    names = names_for(cnt)
    sch = TileScheduler(names, R, C, audit_count=0)
    proc = []
    while True:
        i = sch.next_index()
        if i < 0:
            break
        proc.append(i)
        num = int(names[i].split(".")[0])
        sch.report(i, 0, error=True) if num % 7 == 0 else sch.report(i, cnt[num])
    assert recall(names, cnt, proc)[0] == 0
    print("  检测报错的 tile 按'可能有钻'处理，不漏检")


def test_prescreen_hook():
    cnt = real_counts()
    names = names_for(cnt)
    truth_empty = {i for i, n in enumerate(names) if cnt[int(n.split('.')[0])] == 0}
    sch = TileScheduler(names, R, C, audit_count=0, prescreen=lambda i: i in truth_empty)
    proc = []
    while True:
        i = sch.next_index()
        if i < 0:
            break
        proc.append(i)
        sch.report(i, cnt[int(names[i].split(".")[0])])
    assert len(proc) == 422 and len(sch.skipped_indices()) == 154
    print("  预判钩子（理想预判）：只处理 422 张含钻图，跳过全部 154 张空图")


if __name__ == "__main__":
    for fn in (test_real_disc, test_random_discs_and_shifts, test_center_empty_falls_back,
               test_audit_catches_island, test_disabled_cases, test_error_is_conservative,
               test_prescreen_hook):
        print(fn.__name__)
        fn()
    print("\n全部通过")
