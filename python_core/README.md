# python_core — 算法与部署模块（权威实现）

本目录由 `D:\Develop\defect_detect` **原样迁入**后随 WPF 演进；WPF 仅通过 Bridge 调用。**不要用 C# 重写 softmax / 有效类决策 / 训练逻辑**。

## 应用层类别（2026-08）

- 模型 `class_map.json` / ONNX 输出仍为 **5 类**（含「局部破损」「断钻」）。
- 应用对外有效类 **3 类**：`棱边朝上` / `点朝上` / `面朝上`（`EXCLUDED_CLASSES` 在 `inference_common.py`）。
- `engine.classes` 暴露 3 类；内部 `_model_classes` 保留 5 类以对齐 logits。
- **不再读取** `class_thresholds.json`；`class` / `max_class` / `confidence` 均为有效类内 argmax；`all_scores` 仅含 3 类。

## 模块索引

| 文件 | 职责 |
|------|------|
| `inference_common.py` | 元数据、softmax、有效类决策、批量结果 |
| `inference_engine.py` | 开发版 PyTorch/ONNX 分类 |
| `inference_engine_onnx.py` | 机台 ONNX Runtime 分类 |
| `sahi_detector.py` | SAHI + YOLO 大图流水线 |
| `diamond_uniformity.py` | 单 tile 均匀度评分 + `uniformity_vis` 可视化 |
| `train.py` | 训练 / 微调 / ONNX 导出（训练仍可产生阈值校准文件，应用侧忽略） |
| `analyze_image_sizes.py` | 图像尺寸分析 |
| `app_paths.py` | 路径与 ORT DLL |
| `app_deploy.py` | 原机台入口语义（过渡保留） |
| `scripts/` | 打包与验收 |
| `pyinstaller_hooks/` | 冻结进程 ORT hook |

## 依赖

见 `requirements.txt`（开发）与 `requirements-deploy.txt`（机台）。  
均匀度另需 `scipy`、`shapely`（已写入上述清单）。  
UI 已迁 WPF，**不再依赖 PyQt5**；清单中若仍写 PyQt5，仅为与旧环境对照，新环境可不装。

```powershell
conda activate cv-yolo
pip install -r requirements.txt
```

## 常用命令

```powershell
python analyze_image_sizes.py --data_dir ../data
python train.py --data_dir ../data --img_size 128
python train.py --finetune --extra_data_dirs ../corrections
python train.py --postprocess_only
python scripts/verify_deploy.py
```
