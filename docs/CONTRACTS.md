# 配置与结果契约

WPF 与 `python_core` 必须遵守下列契约，变更时同步更新本文件与操作手册。

## 有效类别（应用层）

| 项 | 约定 |
|----|------|
| 模型输出 | `class_map.json` / ONNX 仍为 5 类：`局部破损`、`断钻`、`棱边朝上`、`点朝上`、`面朝上` |
| 应用有效类 | `棱边朝上`、`点朝上`、`面朝上`（排除前两类） |
| 决策 | 有效类内 softmax argmax；**不使用** `class_thresholds.json` |
| 引擎对外 `Classes` | 仅 3 个有效类（误分类修正按钮、筛选、状态栏类别数） |
| `all_scores` | 仅含 3 个有效类 |

实现：`python_core/inference_common.py`（`EXCLUDED_CLASSES` / `decide_class` / `build_result_dict`）。

## app_config.json

字段名与原 `defect_detect` 一致：

`pt_path`, `onnx_path`, `data_dir`, `corrections_dir`, `use_gpu`, `enable_local_diagnostics`, `yolo_path`, `sahi_device`, `sahi_slice_size`, `sahi_overlap`, `sahi_det_conf`, `sahi_batch_size`, `sahi_crop_padding`, `sahi_output_dir`, `sahi_ios_thresh`, `sahi_min_area_ratio`, `sahi_max_aspect_ratio`, `sahi_edge_filter`, `sahi_edge_margin_px`

- 机台包内路径使用**相对路径**（相对应用根目录）。
- 废弃键 `conf_threshold` 忽略。
- `enable_local_diagnostics`：默认 `false`；为 `true` 时仅写入本机 `logs/diag.jsonl`，不上传。

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

由 `DiamondDetect.Bridge.PythonSahiPipeline.WriteSummaryCsv` 写入输出根目录。

| 列 | 说明 |
|----|------|
| 图像 | 文件名；汇总行为 `批次合计` |
| 汇总钻石数 | 该图检出钻石数；汇总行为各图之和 |
| 棱边朝上 / 点朝上 / 面朝上 | 该图（或批次）各类数量，缺省 0 |
| 检测耗时(s) / 分类耗时(s) / 总耗时(s) | 单图耗时；汇总行留空 |

- **仅处理图像数 > 1** 时追加一行 `批次合计`。
- 单张不写汇总行。
- 补回误删：`scripts/rebuild_summary_from_stats.py`（或打包 exe）扫描 `*/statistics.json` 重建同格式 CSV。

## 环境变量

| 变量 | 含义 |
|------|------|
| `DEFECTS_DEPLOY=1` | 机台模式（ONNX） |
| `DEFECTS_VERIFY=1` | 无 GUI 验收加载 |
| `DIAMOND_PYTHON_HOME` | Conda 环境根目录 |
| `DIAMOND_PYTHON_DLL` | `python3xx.dll` 路径 |

## 管理员口令

与原版一致：`AppSession.AdminPagePassword`（默认 `20250508`）。生产环境请按需修改并避免提交到公开仓库。
