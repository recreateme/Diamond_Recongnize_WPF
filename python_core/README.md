# python_core — 算法与部署模块（权威实现）

本目录由 `D:\Develop\defect_detect` **原样迁入**后随 WPF 演进；WPF 仅通过 Bridge 调用。**不要用 C# 重写 softmax / 有效类决策 / 训练逻辑**。

## 应用层类别（2026-09）

- 真 **3 类**训练与导出：`棱边朝上` / `点朝上` / `面朝上`（骨干 MobileNetV3-Small，Letterbox，默认 `img_size=256`）。
- `EXCLUDED_CLASSES`（`局部破损` / `断钻`）保留以兼容历史 5 类权重；新 3 类模型自动全量生效。
- **不再生成/读取** `class_thresholds.json`；决策为有效类内 argmax。
- 早停默认关闭（`--patience 0`）；再训练页固定传 0，阶段二默认 50 epoch。

## 模块索引

| 文件 | 职责 |
|------|------|
| `inference_common.py` | Letterbox（OpenCV）、BGR 高速预处理、元数据、softmax、有效类决策、批量结果 |
| `model_builder.py` | MobileNetV3-Small 构建（训练 / 开发版推理共用） |
| `inference_engine.py` | 开发版 PyTorch/ONNX 分类 |
| `inference_engine_onnx.py` | 机台 ONNX Runtime 分类 |
| `sahi_detector.py` | SAHI + YOLO 大图流水线；可选 `crop/` 落盘；分类耗时拆预处理/推理 |
| `diamond_uniformity.py` | 单 tile 均匀度评分 + `uniformity_vis` 可视化 |
| `train.py` | MobileNetV3 两阶段训练 / 微调 / ONNX 导出 |
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
python train.py --data_dir ../data --img_size 256 --save_dir ../checkpoints
python train.py --finetune --extra_data_dirs ../corrections --save_dir ../checkpoints
python train.py --postprocess_only --save_dir ../checkpoints
python scripts/verify_deploy.py
```

预训练权重默认路径：`../checkpoints/pretrained/mobilenet_v3_small.pth`（仓库根）。
