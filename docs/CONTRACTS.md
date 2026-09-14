# 配置与结果契约

WPF 与 `python_core` 必须遵守下列契约，变更时同步更新本文件与操作手册。

## 有效类别（应用层）

| 项 | 约定 |
|----|------|
| 模型输出 | 现行 `class_map.json` / ONNX 为 **3 类**：`棱边朝上`、`点朝上`、`面朝上`（EfficientNetV2-S，Letterbox `img_size`） |
| 应用有效类 | 同上三类；加载历史 5 类权重时仍排除「局部破损」「断钻」 |
| 决策 | 有效类内 softmax argmax；**不使用** `class_thresholds.json` |
| 引擎对外 `Classes` | 仅有效类（误分类修正按钮、筛选、状态栏类别数） |
| `all_scores` | 仅含有效类概率 |

实现：`python_core/inference_common.py`（`EXCLUDED_CLASSES` / `decide_class` / `build_result_dict`）。

## app_config.json

字段名与原 `defect_detect` 一致：

`pt_path`, `onnx_path`, `data_dir`, `corrections_dir`, `use_gpu`, `enable_local_diagnostics`, `yolo_path`, `sahi_device`, `sahi_slice_size`, `sahi_overlap`, `sahi_det_conf`, `sahi_batch_size`, `sahi_crop_padding`, `sahi_output_dir`, `sahi_ios_thresh`, `sahi_min_area_ratio`, `sahi_max_aspect_ratio`, `sahi_edge_filter`, `sahi_edge_margin_px`, `uniformity_conf_threshold`

- 机台包内路径使用**相对路径**（相对应用根目录）。
- 废弃键 `conf_threshold` 忽略。
- `enable_local_diagnostics`：默认 `false`；为 `true` 时仅写入本机 `logs/diag.jsonl`，不上传。
- `uniformity_conf_threshold`：均匀度模块 YOLO `det_conf` 过滤阈值，默认 `0.25`。

## 检测结果（DetectionResult）

| 字段 | Python dict 键 | 说明 |
|------|----------------|------|
| Path | `path` | 规范化路径，同路径 upsert |
| Class | `class` | 有效类内预测类别 |
| Confidence | `confidence` | 预测类概率 |
| MaxClass | `max_class` | 与 `class` 相同（有效类 argmax） |
| MaxConfidence | `max_confidence` | 与 `confidence` 相同 |
| AllScores | `all_scores` | 仅有效三类概率 |
| TrueClass | `true_class` | 人工真值 |
| Flagged | `flagged` | 待修正标记 |

## 钻石检测 `summary.csv`

由 `DiamondDetect.Core.Services.SahiSummaryCsv` 写入输出根目录（Bridge 与 ViewModel 在完整模式结束时均会调用，保证单张/多图/文件夹批量一致）。

| 列 | 说明 |
|----|------|
| 图像 | 文件名；汇总行为 `批次合计` |
| 汇总钻石数 | 该图检出钻石数；汇总行为各图之和 |
| 棱边朝上 / 点朝上 / 面朝上 | 该图（或批次）各类数量，缺省 0 |
| 检测耗时(s) | SAHI/YOLO 检测墙钟 |
| 分类预处理(s) | Letterbox 等 CPU 预处理（OpenCV 路径） |
| 分类推理(s) | 模型 forward（batch） |
| 分类耗时(s) | 预处理 + 推理合计（兼容旧列语义） |
| 总耗时(s) | 检测 + 分类；汇总行留空 |

单图耗时列；汇总行（批次合计）时间列留空。

- **仅「检测+分类」完整模式**写入；「仅检测定位」不写（无分类统计）。
- **仅处理图像数 > 1** 时追加一行 `批次合计`。
- 单张不写汇总行。

## 钻石检测每图输出目录

每张图在输出根目录下对应子文件夹 `{stem}/`；**0 目标**时为 `{stem}_无目标/`。

| 文件 | 说明 |
|------|------|
| `detect_boxes.json` / `detect_boxes.csv` | 检测框坐标（与检测用图同分辨率）；含 `det_conf`（YOLO 检测分）；完整模式另含 `defect_class`、`defect_conf`；勾选保存 crop 时含相对路径 `crop_path` |
| `crop/` | 可选：裁剪切片；完整模式为 `crop/<类别>/NNNN.jpg`；仅检测为 `crop/NNNN.jpg` |
| `uniformity_scores.json` | 完整模式且运行均匀度后：单图分数 + 元数据（`status`/`conf_filter` 等） |
| `visualization_classified.jpg` | 完整模式且勾选「保存可视化」且有目标时 |
| `visualization_detection.jpg` | 仅检测模式且勾选「保存可视化」且有目标时 |
| `{stem}_detect_input.jpg` | 仅检测且发生下采样时 |

输出根目录另可有：`summary.csv`（完整模式分类汇总）、`uniformity_summary.csv`（均匀度批量汇总）。

默认不写：`result.json`、`statistics.json`；完整模式不写 `visualization_detection.jpg`。

## 均匀度分析

| 项 | 约定 |
|----|------|
| 实现 | `python_core/diamond_uniformity.py`（Bridge：`IUniformityAnalyzer`） |
| 输入根 | 产品检测输出根（如 576 子图目录）；UI 扫描后勾选子集（默认全选） |
| 认文件 | 每子目录优先 `detect_boxes.json`，否则旧版 `result.json` |
| 排序 | 纯数字子目录名按数值（`1…576`） |
| 过滤 | `det_conf >= uniformity_conf_threshold`；仅三类朝向；无 `det_conf` 时 `conf_filter=skipped` |
| 落盘 | `{子图}/uniformity_scores.json`（覆盖）；根目录 `uniformity_summary.csv`（本次勾选行，覆盖） |
| 可视化 | 可选：`{子图}/uniformity_vis.jpg`（底图+点+凸包+网格密度热力）；点选结果行按需生成；批量默认关、可勾选 |
| 判定 | **只出连续分数**，不做均匀/不均二分类；无盘级汇总分 |

## 环境变量

| 变量 | 含义 |
|------|------|
| `DEFECTS_DEPLOY=1` | 机台模式（ONNX） |
| `DEFECTS_VERIFY=1` | 无 GUI 验收加载 |
| `DIAMOND_PYTHON_HOME` | Conda 环境根目录 |
| `DIAMOND_PYTHON_DLL` | `python3xx.dll` 路径 |

## 管理员口令

与原版一致：`AppSession.AdminPagePassword`（默认 `20250508`）。生产环境请按需修改并避免提交到公开仓库。
