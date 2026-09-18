# 钻石缺陷图像分类系统（WPF）

工业钻石缺陷**三分类**桌面应用：**WPF** + `python_core` 算法。  
现行权重为真 3 类（`棱边朝上` / `点朝上` / `面朝上`，**EfficientNetV2-S**）。当前 `checkpoints/` 为 2026-09-18 均衡 corrections（面:棱边:点=2:2:1）微调正式版：全量 corrections 上 Acc≈**0.83**、Macro-F1≈**0.66**（均衡内 val Macro-F1≈0.75）。详见 [python_core/docs/分类模型现状与下一步.md](python_core/docs/分类模型现状与下一步.md)。加载历史 5 类权重时，应用侧仍排除「局部破损」「断钻」后 argmax。

**功能、算法与模型配置请先读：** [docs/应用功能与算法说明.md](docs/应用功能与算法说明.md)

## 文档索引

| 文档 | 说明 |
|------|------|
| [docs/应用功能与算法说明.md](docs/应用功能与算法说明.md) | **主手册**：七页功能、算法摘要、模型/训练配置 |
| [WPF操作手册.md](WPF操作手册.md) | 开发/运维：环境、调试、维护 |
| [docs/WPF应用开发说明.md](docs/WPF应用开发说明.md) | 页面 / MVVM / Bridge |
| [docs/CONTRACTS.md](docs/CONTRACTS.md) | 配置与结果契约 |
| [docs/打包部署说明.md](docs/打包部署说明.md) | 机台打包与部署 |
| [docs/验收清单.md](docs/验收清单.md) | 功能验收勾选 |
| [docs/钻石均匀度评分模块说明.md](docs/钻石均匀度评分模块说明.md) | 均匀度算法细则 |
| [docs/钻石检测低分辨率漏检优化方案.md](docs/钻石检测低分辨率漏检优化方案.md) | 漏检优化笔记 |
| [python_core/README.md](python_core/README.md) | 算法模块索引 |
| [python_core/docs/分类模型现状与下一步.md](python_core/docs/分类模型现状与下一步.md) | 现行分类指标与精度下一步 |
| [python_core/docs/分类推理速度优化建议.md](python_core/docs/分类推理速度优化建议.md) | 批测瓶颈与下一步速度优化 |

## 快速开始（开发）

```powershell
cd D:\Develop\diamond_detect_wpf
$env:DIAMOND_PYTHON_HOME = "D:\Software\MiniAnaconda\envs\cv-yolo"

dotnet restore
dotnet build
dotnet run --project src\DiamondDetect.Wpf

# 机台模式
$env:DEFECTS_DEPLOY = "1"
dotnet run --project src\DiamondDetect.Wpf

# 无界面验收
.\scripts\verify_wpf.ps1
```

也可用 Visual Studio 打开 `DiamondDetect.sln`，启动项目选 `DiamondDetect.Wpf`。

## 机台打包

```powershell
.\scripts\build_deploy_wpf.ps1 -PythonHome "D:\Software\MiniAnaconda\envs\cv-yolo"
```

产物：`dist\缺陷分类系统\`。详见 [docs/打包部署说明.md](docs/打包部署说明.md)。

## 解决方案结构

| 路径 | 说明 |
|------|------|
| `src/DiamondDetect.Wpf` | 七页 UI（含均匀度分析） |
| `src/DiamondDetect.Core` | 配置、结果模型、接口 |
| `src/DiamondDetect.Bridge` | pythonnet / SAHI / 均匀度 / 训练子进程 |
| `python_core/` | 算法权威实现 |
| `detect_weights/` | YOLO `best.pt`（打包必需） |
| `scripts/` | `publish_wpf` / `build_deploy_wpf` / `verify_wpf` 等 |

## 功能页（一览）

钻石检测分类 · **均匀度分析** · 缺陷检测 · 结果管理 · 误分类修正 · 模型再训练 · 设置  

（检测页输出选项默认全关，可勾选可视化 / crop / 位置文件 / 计算均匀度；独立均匀度页支持历史产品包与可视化。）
