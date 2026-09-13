# -*- coding: utf-8 -*-
"""
diamond_uniformity.py
======================

钻石检测结果 —— 单图(tile)级别均匀度评分 —— 原型/参考实现

【定位】
这是一份"原型/参考实现"脚本：目的是把点模式分析(Voronoi / 最近邻+Clark-Evans /
Delaunay / 网格密度)这几种算法在真实数据上跑通、验证思路是否可行、数值是否符合
直觉。后续会在 C#/WPF 主项目里重新实现一遍，这份脚本主要用来对齐算法逻辑、
作为算法正确性的"标准答案"，不追求工程上的 CLI 封装/异常码规范。

【适用范围（本轮设计已确认的前提）】
1. 分析粒度：单张 tile（子图）级别，不做跨 tile / 整盘坐标合并。
2. 前置过滤：
   - det_conf >= conf_threshold（默认 0.25）
   - defect_class 只保留 {"棱边朝上", "点朝上", "面朝上"} 三类朝向类别，
     "断钻" / "局部破损" 不参与空间位置统计（它们仍然是模型的输出类别，
     只是不代表"钻石在不在这个位置"的可靠位置样本）。
     **例外**：如果这批检测里没有一个点带 defect_class（比如"仅检测定位"
     模式的 detect_boxes.json 压根没有这个字段），类别过滤自动整体跳过，
     直接用全部检测框位置正常计算均匀度——不因为缺一个字段就放弃整批计算，
     也不做额外的警示/标记。
3. 有效分析区域：用检测点(中心点)集合的**凸包**作为该 tile 的有效区域边界，
   而不是整张图像的矩形边界 —— 因为圆盘边缘的 tile 可能只有一个角落有钻石，
   用整图面积会把"图的大部分根本不属于圆盘"误判成"极度稀疏"。
   已知局限：凸包边界处若本该有钻石却漏检了，这类边缘缺陷不会被本方法体系
   捕捉到（凸包天然收缩到已检测点的范围）。这是"只做单图分析、不做整盘坐标
   合并"这个架构决策的代价，后续做整盘合并后应重新评估一次真实圆盘边界。
4. 尺寸差异归一化：钻石表观大小可能因高度/个体差异有 2 倍左右差距。
   每个算法都同时输出 raw（原始中心坐标）和 normalized（用等效半径做归一化，
   把"中心间距"换算成"边到边间隙"或"相对自身尺寸的面积占比"）两个版本。
5. 算法范围：
   - Voronoi 胞元面积 CV（裁剪到凸包内）
   - 最近邻距离 CV + Clark-Evans R
   - Delaunay 边长 CV
   - 网格密度 CV（替代原方案里的"等面积环形分箱"——单图内没有圆盘圆心可以
     参照，所以改成更通用的等分网格，抓"大片区域整体偏疏/偏密"这类问题）
   - Ripley's K/L：留一个可选接口，默认不进主输出（工程上前几种通常够用，
     但保留一个更严谨的候选指标不吃亏）
6. 小样本保护：每个算法都有 min_points 阈值，点数不足时返回 None 并在
   insufficient_data 里标记，而不是硬算一个没有统计意义的数字。
7. 友好百分比（CU / DUlq）：CV/R 这类统计量对非技术用户不直观，额外提供
   Christiansen 均匀系数(CU)和低四分位分布均匀度(DUlq)——业界（灌溉/涂层/
   施肥等均匀性评估）通用的 0~100% 指标，跟原始 CV/R **并列**输出，不互相
   替代。只对 raw 数组算（gap 型归一化值可能为负，不满足 CU/DUlq 假设）。

【依赖】
    pip install numpy scipy shapely

【典型用法】
    from diamond_uniformity import compute_uniformity_scores, load_and_filter

    points = load_and_filter("9.json")               # 读取 + 过滤
    scores = compute_uniformity_scores(points)        # 主输出：{算法名: 分数}
    scores_detail = compute_uniformity_scores(points, return_details=True)

也可以直接 `python diamond_uniformity.py 9.json` 跑一遍看输出；不带参数运行则
用几种合成点集（完美六边形排布 / 完全随机 / 局部聚集+大空洞）做一次自检，
方便你确认这几个指标的相对高低是否符合直觉。
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi, cKDTree, QhullError
from shapely.geometry import Polygon, box as shapely_box
from shapely.ops import unary_union


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #

DEFAULT_ALLOWED_CLASSES = {"棱边朝上", "点朝上", "面朝上"}


@dataclass
class UniformityConfig:
    """所有可调参数集中在这里，方便你之后接标注数据调参。"""

    # ---- 前置过滤 ----
    conf_threshold: float = 0.25
    allowed_classes: set = field(default_factory=lambda: set(DEFAULT_ALLOWED_CLASSES))

    # ---- 小样本保护：低于这个点数，对应算法返回 None ----
    min_points_voronoi: int = 4
    min_points_nn: int = 5
    min_points_delaunay: int = 5
    min_points_grid: int = 4

    # ---- 网格密度法：每个格子期望包含多少个点（用来自适应网格数 n） ----
    grid_target_points_per_cell: float = 8.0
    grid_n_min: int = 2
    grid_n_max: int = 6
    # 网格里被凸包"切"得太碎的格子不参与统计（有效面积占格子面积的比例阈值）
    grid_min_valid_area_ratio: float = 0.15

    # ---- Ripley's K/L：默认不跑，只有显式调用时才算 ----
    run_ripley: bool = False
    ripley_radii: Optional[np.ndarray] = None  # 默认用 hull 尺度自动生成


# --------------------------------------------------------------------------- #
# 数据读取 + 过滤
# --------------------------------------------------------------------------- #

def _normalize_detection(raw: dict) -> Optional[dict]:
    """将 detections[] 或 detect_boxes boxes[] 统一为 cx/cy/w/h + det_conf + defect_class。"""
    if all(k in raw for k in ("cx", "cy", "w", "h")):
        cx = float(raw["cx"])
        cy = float(raw["cy"])
        w = float(raw["w"])
        h = float(raw["h"])
    elif all(k in raw for k in ("x1", "y1", "x2", "y2")):
        x1 = float(raw["x1"])
        y1 = float(raw["y1"])
        x2 = float(raw["x2"])
        y2 = float(raw["y2"])
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        cx = x1 + w * 0.5
        cy = y1 + h * 0.5
    else:
        return None

    has_det_conf = "det_conf" in raw and raw["det_conf"] is not None
    return {
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "det_conf": float(raw["det_conf"]) if has_det_conf else 1.0,
        "defect_class": str(raw.get("defect_class") or ""),
        "_has_det_conf": has_det_conf,
    }


def load_detections(json_path: str) -> list[dict]:
    """读取单张 tile 的检测 JSON。

    兼容：
    - 原型格式：``{"detections": [{cx,cy,w,h,det_conf,defect_class}, ...]}``
    - 当前流水线：``{"boxes": [{x1,y1,x2,y2,det_conf?,defect_class?}, ...]}``
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_list = data.get("detections")
    if raw_list is None:
        raw_list = data.get("boxes") or []

    out: list[dict] = []
    for raw in raw_list:
        if not isinstance(raw, dict):
            continue
        norm = _normalize_detection(raw)
        if norm is not None:
            out.append(norm)
    return out


def filter_detections(
    detections: list[dict],
    config: UniformityConfig = UniformityConfig(),
    *,
    apply_conf_filter: bool = True,
    apply_class_filter: bool = True,
) -> list[dict]:
    """按置信度 + 类别过滤。断钻/局部破损不参与空间位置统计。

    apply_conf_filter=False：旧版 detect_boxes 无 det_conf 时跳过置信度过滤。
    apply_class_filter=False：本批检测完全没有分类信息（如"仅检测定位"模式
        产出的 detect_boxes.json，没有 defect_class 字段）时跳过类别过滤，
        直接按检测框位置正常参与统计——均匀度关心"这个位置有没有钻石"，
        没有朝向分类时用全部检测框仍然是有效信号，不因为缺一个字段就整批
        放弃计算。
    """
    out = []
    for det in detections:
        if apply_conf_filter and float(det.get("det_conf", 0.0)) < config.conf_threshold:
            continue
        if apply_class_filter and det.get("defect_class") not in config.allowed_classes:
            continue
        out.append(det)
    return out


def load_and_filter(json_path: str, config: UniformityConfig = UniformityConfig()) -> list[dict]:
    dets = load_detections(json_path)
    apply_conf = any(d.get("_has_det_conf") for d in dets)
    apply_class = any(str(d.get("defect_class") or "") for d in dets)
    return filter_detections(dets, config, apply_conf_filter=apply_conf, apply_class_filter=apply_class)


def _extract_points_and_sizes(detections: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """从 detections 里拿出中心坐标 (N,2) 和 等效半径 (N,)。

    等效半径用检测框面积换算成"等面积圆"的半径：r = sqrt(w*h/pi)。
    钻石实际形状不是圆，这只是一个近似 —— 如果之后发现归一化后 CV 反而更不
    稳定，可以换成用 (w+h)/4 或对角线的近似方式。
    """
    pts = np.array([[d["cx"], d["cy"]] for d in detections], dtype=float)
    wh = np.array([[d["w"], d["h"]] for d in detections], dtype=float)
    areas = wh[:, 0] * wh[:, 1]
    radii = np.sqrt(np.clip(areas, 1e-9, None) / math.pi)
    return pts, radii


# --------------------------------------------------------------------------- #
# 通用小工具
# --------------------------------------------------------------------------- #

def _cv(values: np.ndarray) -> Optional[float]:
    """变异系数 CV = std / mean。均值接近 0 或数组为空时返回 None。

    注意：normalized（gap-based）系列指标的原始数组可以是负数（比如钻石严重
    拥挤、外接圆互相重叠时 gap<0）。如果这类数组的均值本身是负的，算出来的
    CV 也会是负号 —— 这不是 bug，只是提醒你：gap 型指标的"CV"在均值跨过 0
    附近时已经不太适合直接当"离散程度"解读了，建议这种情况下改看
    return_details 里的原始 gap 数组分布（比如负值占比、最小值）而不是单看
    这一个 CV 数字。
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    mean = values.mean()
    if abs(mean) < 1e-9:
        return None
    return float(values.std(ddof=0) / mean)


def _cu_and_dulq(values: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    """把一批原始测量值（面积/距离/密度，均为非负"量"）压成两个直观的百分比：

    - CU（Christiansen 均匀系数，灌溉/涂层/施肥等工业均匀性评估的通用指标）：
        CU = 100% * (1 - 平均绝对偏差 / 均值)
      100% 代表完全均匀，经验上 >90% 优秀、80~90% 良好、<80% 需要关注。
    - DUlq（低四分位分布均匀度）：
        DUlq = 100% * (数值最低 25% 样本的均值 / 全部样本均值)
      专门盯"最差的一小片区域"，DUlq 越低说明存在越明显的局部稀疏死角；
      全部相等时 DUlq = 100%。

    只对 raw（未做尺寸/gap归一化）的数组算，因为 CU/DUlq 假设的是"一个非负的
    量"（比如水深、涂层厚度），gap 型指标可能是负数，不满足这个前提。
    样本数 < 4 时无法给出有意义的 CU/DUlq，返回 (None, None)。

    极端不均匀（局部空洞导致个别面积/距离远超均值）时 CU 可能算出负数——
    这是公式本身的正常行为，不是bug，但"负的百分比"给人看会很怪，UI 层
    展示时建议 clamp 到 0%（比如 max(0, cu)），不要在这里直接截断，保留
    原始值方便后续做阈值标定时看清楚真实的离散程度。
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 4:
        return None, None
    mean = values.mean()
    if mean <= 1e-9:
        return None, None

    mad = np.mean(np.abs(values - mean))
    cu = 100.0 * (1.0 - mad / mean)

    n_low = max(1, int(math.floor(values.size * 0.25)))
    low_quarter_mean = np.sort(values)[:n_low].mean()
    du_lq = 100.0 * (low_quarter_mean / mean)

    return float(cu), float(du_lq)


def _convex_hull_polygon(points: np.ndarray) -> Optional[Polygon]:
    """点集的凸包，返回 shapely Polygon。点数不足或共线时返回 None。"""
    try:
        hull = ConvexHull(points)
    except QhullError:
        return None
    poly = Polygon(points[hull.vertices])
    if not poly.is_valid or poly.area <= 0:
        return None
    return poly


# --------------------------------------------------------------------------- #
# 方法一：Voronoi 面积 CV
# --------------------------------------------------------------------------- #

def _finite_voronoi_regions(vor: Voronoi, extension: float) -> tuple[list[list[int]], np.ndarray]:
    """把 scipy Voronoi 里的无界胞元补成有界多边形。

    标准做法：对每条指向无穷远的射线（ridge），沿着"两点连线的垂直平分线
    方向"补一个足够远的顶点，让每个输入点都对应一个封闭多边形，返回的
    regions 顺序、下标含义与 vor.points 完全一致（regions[i] 对应第 i 个输入点）。
    补出来的多边形之后会再和凸包做交集裁剪，所以这里补多远不影响最终结果，
    只要"足够远、能完全盖住凸包"就行。
    """
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)

    # 每个点参与的所有 ridge：point_idx -> [(neighbor_idx, v1, v2), ...]
    point_to_ridges: dict[int, list[tuple[int, int, int]]] = {i: [] for i in range(len(vor.points))}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        point_to_ridges[p1].append((p2, v1, v2))
        point_to_ridges[p2].append((p1, v1, v2))

    regions: list[list[int]] = []
    for point_idx in range(len(vor.points)):
        region_idx = vor.point_region[point_idx]
        vertex_ids = vor.regions[region_idx]

        if len(vertex_ids) > 0 and all(v >= 0 for v in vertex_ids):
            regions.append(vertex_ids)
            continue

        # 含有无穷远顶点(-1)，需要补全
        finite_ids = [v for v in vertex_ids if v >= 0]
        for neighbor_idx, v1, v2 in point_to_ridges[point_idx]:
            if v1 >= 0 and v2 >= 0:
                continue  # 有限 ridge，顶点已经在 vertex_ids 里
            finite_vertex = v1 if v2 < 0 else v2
            # 两点连线方向 tangent，垂直方向 normal
            tangent = vor.points[neighbor_idx] - vor.points[point_idx]
            tangent = tangent / (np.linalg.norm(tangent) + 1e-12)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = (vor.points[point_idx] + vor.points[neighbor_idx]) / 2.0
            # normal 的方向要指向"远离两点连线中点、远离整体中心"的一侧
            direction = normal if np.dot(midpoint - center, normal) > 0 else -normal
            far_point = vor.vertices[finite_vertex] + direction * extension
            finite_ids.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # 按极角排序，保证多边形顶点是有序的（否则 shapely 会把它当成自相交）
        verts = np.array([new_vertices[v] for v in finite_ids])
        c = verts.mean(axis=0)
        order = np.argsort(np.arctan2(verts[:, 1] - c[1], verts[:, 0] - c[0]))
        regions.append([finite_ids[i] for i in order])

    return regions, np.asarray(new_vertices)


def voronoi_area_cv(
    points: np.ndarray,
    radii: np.ndarray,
    hull: Polygon,
    min_points: int,
) -> dict:
    """返回 {"raw": CV, "normalized": CV, "areas": ..., "insufficient_data": bool}。

    raw：Voronoi 胞元面积本身的 CV。
    normalized：每个胞元面积 / 该点自身检测框面积 的 CV —— 用来剔除
    "大钻石天然占更大 Voronoi 胞元"这个混淆因素。
    """
    n = len(points)
    if n < min_points or hull is None:
        return {"raw": None, "normalized": None, "areas": None, "insufficient_data": True}

    vor = Voronoi(points)
    extension = max(hull.bounds[2] - hull.bounds[0], hull.bounds[3] - hull.bounds[1]) * 10 + 1.0
    regions, all_vertices = _finite_voronoi_regions(vor, extension)

    areas = np.full(n, np.nan)
    for i, region in enumerate(regions):
        if len(region) < 3:
            continue
        poly = Polygon(all_vertices[region])
        if not poly.is_valid:
            poly = poly.buffer(0)
        clipped = poly.intersection(hull)
        areas[i] = clipped.area

    own_bbox_area = np.pi * radii ** 2  # 与 w*h 等价（见 _extract_points_and_sizes）
    normalized = areas / np.clip(own_bbox_area, 1e-9, None)

    return {
        "raw": _cv(areas),
        "normalized": _cv(normalized),
        "areas": areas,
        "insufficient_data": False,
    }


# --------------------------------------------------------------------------- #
# 方法二：最近邻距离 CV + Clark-Evans R
# --------------------------------------------------------------------------- #

def nearest_neighbor_stats(
    points: np.ndarray,
    radii: np.ndarray,
    hull: Polygon,
    min_points: int,
) -> dict:
    """最近邻距离(raw) + gap(normalized) 的 CV，以及经典 Clark-Evans R（仅 raw 版本）。"""
    n = len(points)
    if n < min_points or hull is None:
        return {
            "nn_cv_raw": None,
            "nn_cv_normalized": None,
            "clark_evans_R": None,
            "nn_distances": None,
            "nn_gaps": None,
            "insufficient_data": True,
        }

    tree = cKDTree(points)
    dist, idx = tree.query(points, k=2)  # k=1 是自己，k=2 才是最近邻
    nn_dist = dist[:, 1]
    nn_idx = idx[:, 1]

    gaps = nn_dist - (radii + radii[nn_idx])  # 边到边间隙，可能为负（说明外接圆重叠）

    density = n / hull.area
    expected_nn = 1.0 / (2.0 * math.sqrt(density))
    clark_evans_r = float(nn_dist.mean() / expected_nn) if expected_nn > 0 else None

    return {
        "nn_cv_raw": _cv(nn_dist),
        "nn_cv_normalized": _cv(gaps),
        "clark_evans_R": clark_evans_r,
        "nn_distances": nn_dist,
        "nn_gaps": gaps,
        "insufficient_data": False,
    }


# --------------------------------------------------------------------------- #
# 方法三：Delaunay 边长 CV
# --------------------------------------------------------------------------- #

def delaunay_edge_cv(
    points: np.ndarray,
    radii: np.ndarray,
    min_points: int,
) -> dict:
    n = len(points)
    if n < min_points:
        return {"raw": None, "normalized": None, "edge_lengths": None, "insufficient_data": True}

    try:
        tri = Delaunay(points)
    except QhullError:
        return {"raw": None, "normalized": None, "edge_lengths": None, "insufficient_data": True}

    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = simplex[i], simplex[(i + 1) % 3]
            edges.add((min(a, b), max(a, b)))

    if not edges:
        return {"raw": None, "normalized": None, "edge_lengths": None, "insufficient_data": True}

    edge_arr = np.array(list(edges))
    diffs = points[edge_arr[:, 0]] - points[edge_arr[:, 1]]
    lengths = np.linalg.norm(diffs, axis=1)
    gaps = lengths - (radii[edge_arr[:, 0]] + radii[edge_arr[:, 1]])

    return {
        "raw": _cv(lengths),
        "normalized": _cv(gaps),
        "edge_lengths": lengths,
        "insufficient_data": False,
    }


# --------------------------------------------------------------------------- #
# 方法四：网格密度 CV（替代"等面积环形分箱"）
# --------------------------------------------------------------------------- #

def grid_density_cv(
    points: np.ndarray,
    hull: Polygon,
    min_points: int,
    target_per_cell: float,
    n_min: int,
    n_max: int,
    min_valid_area_ratio: float,
) -> dict:
    """把凸包的外接矩形切成 n x n 网格，每格用"该格与凸包的交集面积"做分母算
    密度，只保留有效面积占比够高的格子，再算这些格子密度的 CV。

    抓的是"这张图整体一半密一半疏"这种大尺度问题，跟 Voronoi/最近邻抓的
    "点对点局部间距"是互补的两类信号。
    """
    n_points = len(points)
    if n_points < min_points or hull is None:
        return {"cv": None, "n_grid": None, "cell_densities": None, "insufficient_data": True}

    n_grid = int(round(math.sqrt(n_points / target_per_cell)))
    n_grid = max(n_min, min(n_max, n_grid))

    minx, miny, maxx, maxy = hull.bounds
    xs = np.linspace(minx, maxx, n_grid + 1)
    ys = np.linspace(miny, maxy, n_grid + 1)
    cell_w = (maxx - minx) / n_grid
    cell_h = (maxy - miny) / n_grid
    cell_area_nominal = cell_w * cell_h

    densities = []
    for i in range(n_grid):
        for j in range(n_grid):
            cell = shapely_box(xs[i], ys[j], xs[i + 1], ys[j + 1])
            valid_region = cell.intersection(hull)
            valid_area = valid_region.area
            if cell_area_nominal <= 0 or valid_area / cell_area_nominal < min_valid_area_ratio:
                continue  # 被凸包切得太碎的格子不参与统计
            count = int(np.sum(
                (points[:, 0] >= xs[i]) & (points[:, 0] < xs[i + 1] + 1e-9) &
                (points[:, 1] >= ys[j]) & (points[:, 1] < ys[j + 1] + 1e-9)
            ))
            densities.append(count / valid_area)

    if len(densities) < 2:
        return {"cv": None, "n_grid": n_grid, "cell_densities": None, "insufficient_data": True}

    return {
        "cv": _cv(np.array(densities)),
        "n_grid": n_grid,
        "cell_densities": np.array(densities),
        "insufficient_data": False,
    }


def _grid_cells_geometry(
    points: np.ndarray,
    hull: Polygon,
    config: UniformityConfig,
) -> tuple[Optional[int], list[dict]]:
    """与 grid_density_cv 同规则，返回每个有效格子的几何与密度（供可视化）。"""
    n_points = len(points)
    if n_points < config.min_points_grid or hull is None:
        return None, []

    n_grid = int(round(math.sqrt(n_points / config.grid_target_points_per_cell)))
    n_grid = max(config.grid_n_min, min(config.grid_n_max, n_grid))

    minx, miny, maxx, maxy = hull.bounds
    xs = np.linspace(minx, maxx, n_grid + 1)
    ys = np.linspace(miny, maxy, n_grid + 1)
    cell_w = (maxx - minx) / n_grid if n_grid else 0
    cell_h = (maxy - miny) / n_grid if n_grid else 0
    cell_area_nominal = cell_w * cell_h

    cells: list[dict] = []
    for i in range(n_grid):
        for j in range(n_grid):
            x0, x1 = float(xs[i]), float(xs[i + 1])
            y0, y1 = float(ys[j]), float(ys[j + 1])
            cell = shapely_box(x0, y0, x1, y1)
            valid_region = cell.intersection(hull)
            valid_area = valid_region.area
            if cell_area_nominal <= 0 or valid_area / cell_area_nominal < config.grid_min_valid_area_ratio:
                continue
            count = int(np.sum(
                (points[:, 0] >= x0) & (points[:, 0] < x1 + 1e-9) &
                (points[:, 1] >= y0) & (points[:, 1] < y1 + 1e-9)
            ))
            cells.append({
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "density": count / valid_area if valid_area > 0 else 0.0,
            })
    return n_grid, cells


# --------------------------------------------------------------------------- #
# 可视化：底图 + 点 + 凸包 + 网格密度热力
# --------------------------------------------------------------------------- #
# 注意：cv2.putText 用的 Hershey 内置字体不支持中文（会画出一串"?"），下面
# 所有画在图上的文字标签必须用英文/数字/符号，不能用中文——这是踩过的坑，
# 之前的"dense"/"sparse"能显示只是因为凑巧是英文，不代表这里支持中文。真要
# 上中文标签，需要改用 PIL(Pillow) 配中文字体渲染后再贴回 OpenCV 图像。
UNIFORMITY_VIS_NAME = "uniformity_vis.jpg"
_VIS_BG_CANDIDATES = (
    "visualization_classified.jpg",
    "visualization_detection.jpg",
)


def _load_vis_background(tile_dir, payload: Optional[dict], points: np.ndarray):
    """返回 (bgr_image, bg_name_or_empty)。无底图时按 JSON/点集建白底画布。"""
    import cv2
    from pathlib import Path

    d = Path(tile_dir)
    for name in _VIS_BG_CANDIDATES:
        p = d / name
        if p.is_file():
            img = cv2.imread(str(p))
            if img is not None:
                return img, name

    w = h = None
    if isinstance(payload, dict):
        size = payload.get("detect_input_size") or payload.get("image_size") or {}
        if isinstance(size, dict):
            w = int(size.get("width") or 0) or None
            h = int(size.get("height") or 0) or None
    if (w is None or h is None) and len(points) > 0:
        max_x = float(np.max(points[:, 0]))
        max_y = float(np.max(points[:, 1]))
        w = max(64, int(math.ceil(max_x + 20)))
        h = max(64, int(math.ceil(max_y + 20)))
    if w is None or h is None:
        w, h = 1024, 1024
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    return canvas, ""


def _density_to_bgr(t: float) -> tuple[int, int, int]:
    """t in [0,1]: 疏(蓝) → 密(红)，t=0.5 代表"接近均值"。OpenCV BGR。"""
    t = float(np.clip(t, 0.0, 1.0))
    # 简易分三段插值：蓝 -> 青 -> 黄 -> 红
    if t < 0.33:
        u = t / 0.33
        return (255, int(255 * u), 0)
    if t < 0.66:
        u = (t - 0.33) / 0.33
        return (int(255 * (1 - u)), 255, int(255 * u))
    u = (t - 0.66) / 0.34
    return (0, int(255 * (1 - u)), 255)


def _relative_density_color(density: float, mean_density: float, rel_clip: float = 0.75) -> tuple[int, int, int]:
    """按"相对本图平均密度的偏离幅度"上色，而不是本图自己的 min-max。

    之前的实现每张图都单独把 [该图最小密度, 该图最大密度] 拉伸成蓝→红整个
    色域，导致哪怕一张图本身只有 5% 的密度波动，颜色也会被拉成"一半全蓝一半
    全红"，看起来比实际情况严重得多。改成以"偏离均值的相对幅度"定颜色（固定
    ±75% 为色域两端），真正接近均匀的图，颜色会集中在色域中段（不刺眼），
    只有偏离幅度真的很大时才会出现饱和的红/蓝——颜色的"严重程度"才跟真实
    均匀度对得上。
    """
    if mean_density <= 1e-9:
        rel = 0.0
    else:
        rel = (density - mean_density) / mean_density
    t = (float(np.clip(rel, -rel_clip, rel_clip)) + rel_clip) / (2 * rel_clip)
    return _density_to_bgr(t)


def _draw_legend(img, x0: int, y0: int, width: int, height: int, vis_scale: float = 1.0) -> None:
    import cv2

    for i in range(height):
        t = 1.0 - (i / max(1, height - 1))
        color = _density_to_bgr(t)
        cv2.line(img, (x0, y0 + i), (x0 + width - 1, y0 + i), color, 1)
    cv2.rectangle(img, (x0, y0), (x0 + width - 1, y0 + height - 1), (40, 40, 40), 1)
    font_scale = 0.42 * vis_scale
    thick = max(1, int(round(vis_scale)))
    cv2.putText(img, "+75%", (x0 - int(6 * vis_scale), y0 - int(6 * vis_scale)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), thick, cv2.LINE_AA)
    cv2.putText(img, "avg", (x0 - int(6 * vis_scale), y0 + height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), thick, cv2.LINE_AA)
    cv2.putText(img, "-75%", (x0 - int(6 * vis_scale), y0 + height + int(14 * vis_scale)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), thick, cv2.LINE_AA)


def _render_density_panel(
    cells: list[dict],
    hull: Polygon,
    panel_w: int,
    panel_h: int,
    vis_scale: float,
):
    """把网格密度热力图画在独立的一块画布上（不跟检测框/分类框叠在一起）。

    之前的实现是把半透明色块直接盖在已经画了检测框/分类框的底图上，两层
    标注互相打架，格子数又少（最多6x6），看起来容易糊成一片。改成单独一块
    面板，跟主图（底图+点+凸包）左右并排，两个信号都能看清楚。
    """
    import cv2

    img = np.full((panel_h, panel_w, 3), 248, dtype=np.uint8)
    minx, miny, maxx, maxy = hull.bounds
    span_x = (maxx - minx) or 1.0
    span_y = (maxy - miny) or 1.0

    margin = int(round(24 * vis_scale))
    title_h = int(round(26 * vis_scale))
    avail_w = max(1, panel_w - 2 * margin)
    avail_h = max(1, panel_h - 2 * margin - title_h)
    scale = min(avail_w / span_x, avail_h / span_y)
    offset_x = margin + (avail_w - span_x * scale) / 2.0
    offset_y = margin + title_h + (avail_h - span_y * scale) / 2.0

    def to_panel_xy(x: float, y: float) -> tuple[int, int]:
        return (
            int(round(offset_x + (x - minx) * scale)),
            int(round(offset_y + (y - miny) * scale)),
        )

    if cells:
        dens = np.array([c["density"] for c in cells], dtype=float)
        mean_density = float(dens.mean())
        for c in cells:
            color = _relative_density_color(c["density"], mean_density)
            x0, y0 = to_panel_xy(c["x0"], c["y0"])
            x1, y1 = to_panel_xy(c["x1"], c["y1"])
            if x1 > x0 and y1 > y0:
                cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
                cv2.rectangle(img, (x0, y0), (x1, y1), (90, 90, 90), max(1, int(round(vis_scale))))

    hx, hy = hull.exterior.xy
    hull_pts = np.array([to_panel_xy(x, y) for x, y in zip(hx, hy)], dtype=np.int32)
    if len(hull_pts) >= 2:
        cv2.polylines(img, [hull_pts], isClosed=True, color=(0, 150, 0),
                      thickness=max(1, int(round(vis_scale))))

    cv2.putText(img, "Density (rel. to avg)", (margin, int(round(18 * vis_scale))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * vis_scale, (30, 30, 30),
                max(1, int(round(vis_scale))), cv2.LINE_AA)

    legend_w = max(12, int(round(14 * vis_scale)))
    legend_h = max(60, int(round(panel_h * 0.26)))
    lx = max(margin, panel_w - legend_w - int(round(26 * vis_scale)))
    ly = panel_h - legend_h - int(round(24 * vis_scale))
    _draw_legend(img, lx, ly, legend_w, legend_h, vis_scale)
    return img


def render_uniformity_visualization(
    json_path: str,
    conf_threshold: float = 0.25,
    out_path: Optional[str] = None,
) -> dict:
    """生成 uniformity_vis.jpg：左侧底图+点+凸包，右侧独立的网格密度面板。"""
    import cv2
    from pathlib import Path

    path = Path(json_path)
    tile_dir = path.parent
    vis_path = Path(out_path) if out_path else (tile_dir / UNIFORMITY_VIS_NAME)
    config = UniformityConfig(conf_threshold=float(conf_threshold))

    result = {
        "vis_path": "",
        "status": "ok",
        "background": "",
        "n_points": 0,
        "grid_n": None,
    }

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        dets = load_detections(str(path))
        apply_conf = any(d.get("_has_det_conf") for d in dets)
        apply_class = any(str(d.get("defect_class") or "") for d in dets)
        filtered = filter_detections(dets, config, apply_conf_filter=apply_conf, apply_class_filter=apply_class)
        result["n_points"] = len(filtered)
        if len(filtered) < 3:
            result["status"] = "跳过:点数不足"
            return result

        points, _radii = _extract_points_and_sizes(filtered)
        hull = _convex_hull_polygon(points)
        if hull is None:
            result["status"] = "跳过:无法建凸包"
            return result

        base, bg_name = _load_vis_background(tile_dir, payload if isinstance(payload, dict) else None, points)
        result["background"] = bg_name
        main_img = base.copy()
        h, w = main_img.shape[:2]
        # 标注元素尺寸系数：以 960px 为参考基准。原图分辨率越高（比如5120px的
        # tile），点/线/字就按比例放大，避免最终缩到预览缩略图（~720px）里时
        # 小到看不清——之前是按固定像素数画的，只在原图原样查看时看得清楚。
        vis_scale = max(1.0, min(w, h) / 960.0)

        n_grid, cells = _grid_cells_geometry(points, hull, config)
        result["grid_n"] = n_grid

        # 主图只画点 + 凸包，不叠加密度色块——避免跟底图上已有的检测框/分类框
        # 打架糊成一片；密度信号改到右侧独立面板里展示。
        hx, hy = hull.exterior.xy
        hull_pts = np.array([[int(round(x)), int(round(y))] for x, y in zip(hx, hy)], dtype=np.int32)
        if len(hull_pts) >= 2:
            cv2.polylines(main_img, [hull_pts], isClosed=True, color=(0, 180, 0),
                          thickness=max(2, int(round(2 * vis_scale))))

        r = max(3, int(round((min(w, h) / 400) * vis_scale)))
        for x, y in points:
            cx, cy = int(round(x)), int(round(y))
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(main_img, (cx, cy), r, (0, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(main_img, (cx, cy), r, (0, 120, 120), max(1, int(round(vis_scale))), lineType=cv2.LINE_AA)

        label = f"n={len(filtered)}"
        font_scale = 0.8 * vis_scale
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                 max(2, int(round(2 * vis_scale))))
        pad = int(round(6 * vis_scale))
        origin = (int(round(12 * vis_scale)), int(round(28 * vis_scale)))
        # 先垫一块半透明底色再写字，避免文字压在黄色标注点/浅色底图上看不清
        box_overlay = main_img.copy()
        cv2.rectangle(box_overlay, (origin[0] - pad, origin[1] - label_h - pad),
                      (origin[0] + label_w + pad, origin[1] + pad), (0, 0, 0), -1)
        main_img = cv2.addWeighted(box_overlay, 0.45, main_img, 0.55, 0)
        cv2.putText(main_img, label, origin,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), max(1, int(round(vis_scale))), cv2.LINE_AA)

        panel_w = max(int(round(220 * vis_scale)), int(round(w * 0.32)))
        panel = _render_density_panel(cells, hull, panel_w, h, vis_scale)

        gap = max(4, int(round(6 * vis_scale)))
        combined = np.full((h, w + gap + panel_w, 3), 255, dtype=np.uint8)
        combined[:, :w] = main_img
        combined[:, w + gap:] = panel

        cv2.imwrite(str(vis_path), combined, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        result["vis_path"] = str(vis_path.resolve())
    except Exception as ex:  # noqa: BLE001
        result["status"] = f"可视化失败 ({type(ex).__name__})"
        result["error"] = str(ex)

    return result


# --------------------------------------------------------------------------- #
# 方法五（可选）：Ripley's K / L
# --------------------------------------------------------------------------- #

def ripley_l_function(points: np.ndarray, hull: Polygon, radii: Optional[np.ndarray] = None) -> dict:
    """简化版 Ripley's L 函数，不做边界校正（工程上默认不启用，只是留个接口）。

    L(r) - r > 0 说明在尺度 r 上比随机分布更聚集；
    L(r) - r < 0 说明在尺度 r 上比随机分布更规则/均匀；
    L(r) - r ≈ 0 说明接近完全随机分布(CSR)。

    没做边界校正，意味着靠近凸包边界的点，其邻居会被系统性低估 —— 数值仅供
    参考，不建议直接拿来定阈值。
    """
    n = len(points)
    if n < 10 or hull is None:
        return {"radii": None, "L": None, "insufficient_data": True}

    area = hull.area
    if radii is None:
        max_r = max(hull.bounds[2] - hull.bounds[0], hull.bounds[3] - hull.bounds[1]) / 4
        radii = np.linspace(max_r / 10, max_r, 10)

    tree = cKDTree(points)
    l_values = []
    for r in radii:
        pairs = tree.query_pairs(r)
        k_r = (2 * len(pairs) * area) / (n * (n - 1))
        l_values.append(math.sqrt(k_r / math.pi))

    return {"radii": radii, "L": np.array(l_values), "insufficient_data": False}


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #

def compute_uniformity_scores(
    detections: list[dict],
    config: UniformityConfig = UniformityConfig(),
    return_details: bool = False,
    already_filtered: bool = True,
) -> dict:
    """输入一份 detections 列表，输出各算法的均匀度分数。

    detections: 每个元素至少要有 cx, cy, w, h 字段（跟你给的检测 JSON 格式一致）。
        如果 already_filtered=False，会先按 config 里的置信度/类别规则过滤一遍；
        如果你在调用前已经自己过滤好了（比如批处理时提前统一过滤），可以把
        already_filtered 设为 True（默认）跳过重复过滤。
    return_details: True 时额外返回每个算法的中间数组（Voronoi 各胞元面积、
        最近邻距离数组等），供之后做可视化/直方图用，不需要重新跑一遍算法。
    """
    if not already_filtered:
        detections = filter_detections(detections, config)

    n_total = len(detections)
    result: dict = {
        "n_points": n_total,
        "boundary_method": "convex_hull",
    }

    if n_total < 3:
        result.update({
            "hull_area": None,
            "voronoi_area_cv_raw": None,
            "voronoi_area_cv_normalized": None,
            "nn_distance_cv_raw": None,
            "nn_distance_cv_normalized": None,
            "clark_evans_R": None,
            "delaunay_edge_cv_raw": None,
            "delaunay_edge_cv_normalized": None,
            "grid_density_cv": None,
            "grid_n": None,
            "voronoi_area_cu": None,
            "voronoi_area_du_lq": None,
            "nn_distance_cu": None,
            "nn_distance_du_lq": None,
            "delaunay_edge_cu": None,
            "delaunay_edge_du_lq": None,
            "grid_density_cu": None,
            "grid_density_du_lq": None,
            "insufficient_data": {
                "voronoi": True, "nn": True, "delaunay": True, "grid": True,
            },
        })
        if return_details:
            result["details"] = {}
        return result

    points, radii = _extract_points_and_sizes(detections)
    hull = _convex_hull_polygon(points)

    vor_res = voronoi_area_cv(points, radii, hull, config.min_points_voronoi)
    nn_res = nearest_neighbor_stats(points, radii, hull, config.min_points_nn)
    del_res = delaunay_edge_cv(points, radii, config.min_points_delaunay)
    grid_res = grid_density_cv(
        points, hull, config.min_points_grid,
        config.grid_target_points_per_cell, config.grid_n_min, config.grid_n_max,
        config.grid_min_valid_area_ratio,
    )

    # 友好百分比指标（CU / DUlq）：并列于 CV/R 之外展示，供UI/CSV直接给非技术
    # 用户看；只用 raw 数组算（gap 型归一化值可能为负，不满足 CU/DUlq 的假设）。
    voronoi_cu, voronoi_du = _cu_and_dulq(vor_res["areas"]) if vor_res["areas"] is not None else (None, None)
    nn_cu, nn_du = _cu_and_dulq(nn_res["nn_distances"]) if nn_res["nn_distances"] is not None else (None, None)
    del_cu, del_du = _cu_and_dulq(del_res["edge_lengths"]) if del_res["edge_lengths"] is not None else (None, None)
    grid_cu, grid_du = _cu_and_dulq(grid_res["cell_densities"]) if grid_res["cell_densities"] is not None else (None, None)

    result.update({
        "hull_area": hull.area if hull is not None else None,
        "voronoi_area_cv_raw": vor_res["raw"],
        "voronoi_area_cv_normalized": vor_res["normalized"],
        "nn_distance_cv_raw": nn_res["nn_cv_raw"],
        "nn_distance_cv_normalized": nn_res["nn_cv_normalized"],
        "clark_evans_R": nn_res["clark_evans_R"],
        "delaunay_edge_cv_raw": del_res["raw"],
        "delaunay_edge_cv_normalized": del_res["normalized"],
        "grid_density_cv": grid_res["cv"],
        "grid_n": grid_res["n_grid"],
        "voronoi_area_cu": voronoi_cu,
        "voronoi_area_du_lq": voronoi_du,
        "nn_distance_cu": nn_cu,
        "nn_distance_du_lq": nn_du,
        "delaunay_edge_cu": del_cu,
        "delaunay_edge_du_lq": del_du,
        "grid_density_cu": grid_cu,
        "grid_density_du_lq": grid_du,
        "insufficient_data": {
            "voronoi": vor_res["insufficient_data"],
            "nn": nn_res["insufficient_data"],
            "delaunay": del_res["insufficient_data"],
            "grid": grid_res["insufficient_data"],
        },
    })

    if config.run_ripley:
        ripley_res = ripley_l_function(points, hull, config.ripley_radii)
        result["ripley_L"] = ripley_res["L"].tolist() if ripley_res["L"] is not None else None
        result["ripley_radii"] = ripley_res["radii"].tolist() if ripley_res["radii"] is not None else None

    if return_details:
        result["details"] = {
            "points": points.tolist(),
            "radii": radii.tolist(),
            "voronoi_cell_areas": vor_res["areas"].tolist() if vor_res["areas"] is not None else None,
            "nn_distances": nn_res["nn_distances"].tolist() if nn_res["nn_distances"] is not None else None,
            "nn_gaps": nn_res["nn_gaps"].tolist() if nn_res["nn_gaps"] is not None else None,
            "delaunay_edge_lengths": del_res["edge_lengths"].tolist() if del_res["edge_lengths"] is not None else None,
            "grid_cell_densities": grid_res["cell_densities"].tolist() if grid_res["cell_densities"] is not None else None,
        }

    return result


# --------------------------------------------------------------------------- #
# 自检 / demo：合成三种典型点分布，看指标是否符合直觉
# --------------------------------------------------------------------------- #

def _make_hex_grid(n_side: int, spacing: float = 100.0, jitter: float = 0.0, seed: int = 0) -> np.ndarray:
    """完美六边形排布（+可选轻微抖动），代表"均匀"的极端情况。"""
    rng = np.random.default_rng(seed)
    pts = []
    for row in range(n_side):
        offset = (spacing / 2) if row % 2 else 0.0
        for col in range(n_side):
            x = col * spacing + offset
            y = row * spacing * math.sqrt(3) / 2
            pts.append([x, y])
    pts = np.array(pts, dtype=float)
    if jitter > 0:
        pts += rng.normal(0, jitter, size=pts.shape)
    return pts


def _make_random(n: int, extent: float = 1000.0, seed: int = 1) -> np.ndarray:
    """完全随机分布(CSR)，代表"中间态"。"""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, extent, size=(n, 2))


def _make_clustered_with_hole(n: int, extent: float = 1000.0, seed: int = 2) -> np.ndarray:
    """局部聚集 + 一块明显空洞，代表"不均匀"的典型缺陷模式。"""
    rng = np.random.default_rng(seed)
    pts = []
    n_clusters = 4
    per_cluster = n // n_clusters
    centers = rng.uniform(extent * 0.15, extent * 0.85, size=(n_clusters, 2))
    for c in centers:
        pts.append(rng.normal(c, extent * 0.04, size=(per_cluster, 2)))
    pts = np.concatenate(pts, axis=0)
    # 挖掉右上角一块区域模拟漏检空洞
    mask = ~((pts[:, 0] > extent * 0.6) & (pts[:, 1] > extent * 0.6))
    return pts[mask]


def _points_to_detections(points: np.ndarray, base_size: float = 30.0, size_jitter: float = 0.0, seed: int = 3) -> list[dict]:
    """把裸坐标包装成 detections 格式，方便复用同一套 compute_uniformity_scores。"""
    rng = np.random.default_rng(seed)
    sizes = base_size * (1 + rng.uniform(-size_jitter, size_jitter, size=len(points)))
    dets = []
    for (x, y), s in zip(points, sizes):
        dets.append({
            "cx": float(x), "cy": float(y), "w": float(s), "h": float(s),
            "det_conf": 0.9, "defect_class": "面朝上",
        })
    return dets


def _self_test() -> None:
    print("未提供 JSON 路径，改跑合成数据自检（六边形排布 / 完全随机 / 聚集+空洞）：\n")
    scenarios = {
        "完美六边形排布（应最均匀）": _points_to_detections(_make_hex_grid(8, spacing=100)),
        "完全随机分布（中间态）": _points_to_detections(_make_random(64)),
        "局部聚集+大空洞（应最不均匀）": _points_to_detections(_make_clustered_with_hole(80)),
        "六边形排布+尺寸差2倍（检验尺寸归一化是否生效）":
            _points_to_detections(_make_hex_grid(8, spacing=100), size_jitter=0.5),
    }
    for name, dets in scenarios.items():
        scores = compute_uniformity_scores(dets, already_filtered=True)
        print(f"--- {name} (n={scores['n_points']}) ---")
        for k in [
            "voronoi_area_cv_raw", "voronoi_area_cv_normalized", "voronoi_area_cu", "voronoi_area_du_lq",
            "nn_distance_cv_raw", "nn_distance_cv_normalized", "clark_evans_R", "nn_distance_cu", "nn_distance_du_lq",
            "delaunay_edge_cv_raw", "delaunay_edge_cv_normalized", "delaunay_edge_cu", "delaunay_edge_du_lq",
            "grid_density_cv", "grid_n", "grid_density_cu", "grid_density_du_lq",
        ]:
            v = scores[k]
            print(f"  {k:32s}: {v:.4f}" if isinstance(v, float) else f"  {k:32s}: {v}")
        print()
    print(
        "预期看到的规律：\n"
        "  1) CV/R 越接近'均匀'方向的场景，Voronoi/最近邻/Delaunay 的 CV 越低、\n"
        "     Clark-Evans R 越大（>1 代表比随机更规则）；聚集+空洞场景相反。\n"
        "  2) 最后一个场景（六边形+尺寸抖动）里，如果 normalized 版本的 CV 明显\n"
        "     低于 raw 版本，说明尺寸归一化确实在剔除'纯尺寸差异'带来的干扰。\n"
        "  3) CU/DUlq 是 0~100% 的友好指标，跟第1条同方向：越'均匀'的场景数值\n"
        "     应越接近100%；DUlq 专盯最差25%区域，聚集+空洞场景应明显偏低。\n"
    )


# --------------------------------------------------------------------------- #
# 工程入口（WPF Bridge / 批量落盘）
# --------------------------------------------------------------------------- #

SCORE_KEYS = (
    "n_points",
    "boundary_method",
    "hull_area",
    "voronoi_area_cv_raw",
    "voronoi_area_cv_normalized",
    "nn_distance_cv_raw",
    "nn_distance_cv_normalized",
    "clark_evans_R",
    "delaunay_edge_cv_raw",
    "delaunay_edge_cv_normalized",
    "grid_density_cv",
    "grid_n",
    "voronoi_area_cu",
    "voronoi_area_du_lq",
    "nn_distance_cu",
    "nn_distance_du_lq",
    "delaunay_edge_cu",
    "delaunay_edge_du_lq",
    "grid_density_cu",
    "grid_density_du_lq",
    "insufficient_data",
)

# CSV 表头改中文（原来的英文字段名跟项目其它 CSV 的风格不一致）；CU/DU 友好
# 百分比与原始 CV/R 并列展示，不互相替代——前者给人看，后者给后续人工标注
# + 自动定阈值用。每项是 (内部字段名, CSV表头文字)。
SUMMARY_COLUMNS = (
    ("image", "图像"),
    ("n_points", "钻石数"),
    ("voronoi_area_cu", "Voronoi均匀度%"),
    ("voronoi_area_cv_normalized", "Voronoi面积CV"),
    ("nn_distance_cu", "间距均匀度%"),
    ("clark_evans_R", "规则度R"),
    ("delaunay_edge_cu", "三角网均匀度%"),
    ("delaunay_edge_cv_normalized", "Delaunay边长CV"),
    ("grid_density_cu", "区域均匀度%"),
    ("grid_density_du_lq", "最差区域均匀度%"),
    ("grid_density_cv", "区域密度CV"),
    ("status", "状态"),
    ("conf_filter", "置信度过滤"),
)

_SUMMARY_TEXT_KEYS = {"image", "n_points", "status", "conf_filter"}


def _fmt_summary_value(value, key: str) -> str:
    """CSV 单元格格式化：文本类字段原样输出，数值类字段走 _fmt_score。"""
    if key in _SUMMARY_TEXT_KEYS:
        return "" if value is None else str(value)
    return _fmt_score(value)


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _fmt_score(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def analyze_detect_boxes_file(
    json_path: str,
    conf_threshold: float = 0.25,
    write_scores: bool = True,
    write_vis: bool = False,
) -> dict:
    """分析单个 detect_boxes.json / result.json（或原型 detections JSON），可选写出 uniformity_scores.json。"""
    from pathlib import Path

    path = Path(json_path)
    # 产品包按子目录名标识 tile（如 defects/11）；勿被 JSON 内 image 文件名覆盖
    image_name = path.parent.name if path.name.lower() in {
        "detect_boxes.json", "result.json",
    } else path.stem
    source_json = str(path.resolve())
    config = UniformityConfig(conf_threshold=float(conf_threshold))

    result: dict = {
        "image": image_name,
        "source_json": source_json,
        "source_kind": path.name,
        "conf_threshold": float(conf_threshold),
        "conf_filter": "skipped",
        "status": "ok",
        "scores_json": "",
        "vis_path": "",
        "n_points": 0,
    }

    def _maybe_write() -> None:
        if not write_scores:
            return
        out_path = path.parent / "uniformity_scores.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        result["scores_json"] = str(out_path.resolve())

    def _maybe_vis() -> None:
        if not write_vis:
            return
        vis = render_uniformity_visualization(str(path), conf_threshold=conf_threshold)
        result["vis_path"] = vis.get("vis_path") or ""
        if vis.get("status") and vis["status"] != "ok" and not result["vis_path"]:
            result["vis_status"] = vis["status"]

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if (
            path.name.lower() not in {"detect_boxes.json", "result.json"}
            and isinstance(payload, dict)
            and payload.get("image")
        ):
            result["image"] = str(payload["image"])

        dets = load_detections(str(path))
        if not dets:
            result["status"] = "跳过:无检测框"
            _maybe_write()
            return result

        apply_conf = any(d.get("_has_det_conf") for d in dets)
        # 「仅检测定位」模式没有 defect_class 字段：不因此放弃计算，退化为只按
        # 检测框位置统计（不做朝向类别过滤）。
        apply_class = any(str(d.get("defect_class") or "") for d in dets)
        result["conf_filter"] = "applied" if apply_conf else "skipped"
        filtered = filter_detections(dets, config, apply_conf_filter=apply_conf, apply_class_filter=apply_class)
        scores = compute_uniformity_scores(filtered, config, already_filtered=True)
        for key in SCORE_KEYS:
            if key in scores:
                result[key] = _json_safe(scores[key])

        insuff = scores.get("insufficient_data") or {}
        if filtered and isinstance(insuff, dict) and all(insuff.values()):
            result["status"] = "样本不足"
        elif not filtered:
            result["status"] = "跳过:过滤后无点"
            result["n_points"] = 0

        _maybe_write()
        if result.get("n_points", 0) >= 3:
            _maybe_vis()
    except Exception as ex:  # noqa: BLE001 — 批量时单张失败继续
        result["status"] = f"跳过:JSON损坏 ({type(ex).__name__})"
        result["error"] = str(ex)
        result["n_points"] = 0
        try:
            _maybe_write()
        except Exception:  # noqa: BLE001
            pass

    return result


def write_uniformity_summary_csv(output_root: str, rows: list[dict]) -> str:
    """写出输出根目录 uniformity_summary.csv（中文表头；CU/DU友好百分比与原始CV/R并列）。"""
    import csv
    from pathlib import Path

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "uniformity_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([label for _, label in SUMMARY_COLUMNS])
        for row in rows:
            writer.writerow([_fmt_summary_value(row.get(key), key) for key, _ in SUMMARY_COLUMNS])
    return str(csv_path.resolve())


def _tile_sort_key(name: str):
    """纯数字名按数值排（1…576），其余按字符串。"""
    s = str(name)
    if s.isdigit():
        return (0, int(s), "")
    return (1, 0, s.lower())


def resolve_tile_json(tile_dir) -> Optional[str]:
    """子目录内优先 detect_boxes.json，否则 result.json。"""
    from pathlib import Path

    d = Path(tile_dir)
    boxes = d / "detect_boxes.json"
    if boxes.is_file():
        return str(boxes.resolve())
    result = d / "result.json"
    if result.is_file():
        return str(result.resolve())
    return None


def list_tile_jsons(output_root: str) -> list[dict]:
    """列出产品输出根下可分析的 tile JSON（已按数字名排序）。"""
    from pathlib import Path

    root = Path(output_root)
    if not root.is_dir():
        raise FileNotFoundError(f"输出目录不存在: {output_root}")

    items: list[dict] = []
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        jp = resolve_tile_json(sub)
        if not jp:
            continue
        kind = Path(jp).name
        items.append({
            "image": sub.name,
            "json_path": jp,
            "source_kind": kind,
        })

    # 根目录自身就是单图输出时
    root_jp = resolve_tile_json(root)
    if root_jp and not any(i["json_path"] == root_jp for i in items):
        # 仅当根下没有子目录 tile 时采用根级 JSON，避免把「产品根」误当成一张图
        if not items:
            items.append({
                "image": root.name,
                "json_path": root_jp,
                "source_kind": Path(root_jp).name,
            })

    items.sort(key=lambda x: _tile_sort_key(x["image"]))
    return items


def analyze_output_root(
    output_root: str,
    conf_threshold: float = 0.25,
    write_scores: bool = True,
    write_summary: bool = True,
    json_paths: Optional[list] = None,
    write_vis: bool = False,
) -> dict:
    """扫描或按给定 json_paths 分析；写回各子目录 scores + 根目录 summary。

    json_paths: 若提供则只分析这些文件（子集/单张）；否则分析根下全部 tile。
    write_vis: True 时同时写出 uniformity_vis.jpg（默认关）。
    """
    from pathlib import Path

    root = Path(output_root)
    if not root.is_dir():
        raise FileNotFoundError(f"输出目录不存在: {output_root}")

    if json_paths:
        paths = [Path(p) for p in json_paths]
    else:
        paths = [Path(item["json_path"]) for item in list_tile_jsons(str(root))]

    rows: list[dict] = []
    for jp in paths:
        rows.append(analyze_detect_boxes_file(
            str(jp),
            conf_threshold=conf_threshold,
            write_scores=write_scores,
            write_vis=write_vis,
        ))

    summary_path = ""
    if write_summary and rows:
        summary_path = write_uniformity_summary_csv(str(root), rows)

    return {
        "output_root": str(root.resolve()),
        "count": len(rows),
        "summary_csv": summary_path,
        "rows": rows,
    }


def main() -> None:
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        config = UniformityConfig()
        filtered = load_and_filter(json_path, config)
        print(f"读取 {json_path}：原始 {len(load_detections(json_path))} 个检测，"
              f"过滤后 {len(filtered)} 个参与均匀度统计。\n")
        scores = compute_uniformity_scores(filtered, config, return_details=False)
        print(json.dumps(scores, ensure_ascii=False, indent=2))
    else:
        _self_test()


if __name__ == "__main__":
    main()
