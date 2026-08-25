# python_core — 算法与部署模块（权威实现）

本目录由 `D:\Develop\defect_detect` **原样迁入**，WPF 仅通过 Bridge 调用，**不要用 C# 重写阈值/softmax/训练逻辑**。

## 模块索引

| 文件 | 职责 |
|------|------|
| `inference_common.py` | 元数据、softmax、阈值、批量结果 |
| `inference_engine.py` | 开发版 PyTorch/ONNX 分类 |
| `inference_engine_onnx.py` | 机台 ONNX Runtime 分类 |
| `sahi_detector.py` | SAHI + YOLO 大图流水线 |
| `train.py` | 训练 / 微调 / ONNX 导出 |
| `analyze_image_sizes.py` | 图像尺寸分析 |
| `app_paths.py` | 路径与 ORT DLL |
| `app_deploy.py` | 原机台入口语义（过渡保留） |
| `scripts/` | 打包与验收 |
| `pyinstaller_hooks/` | 冻结进程 ORT hook |

## 依赖

见 `requirements.txt`（开发）与 `requirements-deploy.txt`（机台）。  
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
