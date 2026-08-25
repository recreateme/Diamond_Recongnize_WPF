# 配置与结果契约

WPF 与 `python_core` 必须遵守下列契约，变更时同步更新本文件与操作手册。

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
| Class | `class` | 阈值决策后类别 |
| Confidence | `confidence` | 预测类概率 |
| MaxClass | `max_class` | argmax 类 |
| MaxConfidence | `max_confidence` | argmax 概率 |
| AllScores | `all_scores` | 各类概率 |
| TrueClass | `true_class` | 人工真值 |
| Flagged | `flagged` | 待修正标记 |

## 环境变量

| 变量 | 含义 |
|------|------|
| `DEFECTS_DEPLOY=1` | 机台模式（ONNX） |
| `DEFECTS_VERIFY=1` | 无 GUI 验收加载 |
| `DIAMOND_PYTHON_HOME` | Conda 环境根目录 |
| `DIAMOND_PYTHON_DLL` | `python3xx.dll` 路径 |

## 管理员口令

与原版一致：`AppSession.AdminPagePassword`（默认 `20250508`）。生产环境请按需修改并避免提交到公开仓库。
