# 钻石缺陷图像分类系统（WPF 重构版）

工业钻石缺陷**三分类**（应用层）：**WPF 桌面界面** + 保留原 Python 算法（`python_core/`）。

有效类别：`棱边朝上` / `点朝上` / `面朝上`。ONNX 模型仍为 5 类输出，推理时排除「局部破损」「断钻」，在有效类中取 argmax；**不再使用** `class_thresholds.json`。

| 文档 | 说明 |
|------|------|
| [WPF操作手册.md](WPF操作手册.md) | 技术栈、模块、开发与维护（**新手先读**） |
| [docs/WPF应用开发说明.md](docs/WPF应用开发说明.md) | 页面/MVVM/Bridge 开发指南 |
| [docs/打包部署说明.md](docs/打包部署说明.md) | 机台打包与部署 |
| [docs/验收清单.md](docs/验收清单.md) | 功能验收勾选表 |
| [docs/CONTRACTS.md](docs/CONTRACTS.md) | 配置与结果契约（含 `summary.csv`、每图输出目录） |
| [docs/钻石检测低分辨率漏检优化方案.md](docs/钻石检测低分辨率漏检优化方案.md) | 下采样/压缩图漏检分析与后续优化路线 |
| [WPF重构实施计划.md](WPF重构实施计划.md) | 分阶段计划与进度 |

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

也可用 **Visual Studio 2026** 打开 `DiamondDetect.sln`，启动项目选 `DiamondDetect.Wpf`。

## 机台打包

```powershell
# 完整包（含 Python 运行时，体积大）
.\scripts\build_deploy_wpf.ps1 -PythonHome "D:\Software\MiniAnaconda\envs\cv-yolo"

# 瘦包（机台自备 Conda）
.\scripts\build_deploy_wpf.ps1 -SkipPythonRuntime
```

产物：`dist\缺陷分类系统\`。详见 [docs/打包部署说明.md](docs/打包部署说明.md)。

## 解决方案结构

| 路径 | 说明 |
|------|------|
| `src/DiamondDetect.Wpf` | 六大功能页 UI |
| `src/DiamondDetect.Core` | 配置、结果模型、接口 |
| `src/DiamondDetect.Bridge` | pythonnet / SAHI / 训练子进程 |
| `python_core/` | 原算法权威实现（勿用 C# 重写 softmax / 有效类决策） |
| `detect_weights/` | YOLO 权重目录（`best.pt` 本地放置，见 README；打包必需） |
| `scripts/` | `publish_wpf` / `build_deploy_wpf` / `verify_wpf`；辅助脚本见下表 |

### 辅助脚本（`scripts/`）

| 脚本 | 用途 |
|------|------|
| `export_boxes_from_result.py` | 从旧版 `result.json` 导出 OpenCV 坐标（兼容迁移） |
| `test_vis_first_box.py` | 单框可视化调试 |

## 近期变更摘要（2026-09）

- **钻石检测页**：可选「仅检测定位」+ 下采样；「保存可视化」；协作式停止；Grid 布局优化结果表。
- **完整模式输出**：每图 `detect_boxes.json/csv`（含分类列）；可选 `visualization_classified.jpg`；根目录 `summary.csv`（`SahiSummaryCsv`，单张/文件夹均支持）。
- **仅检测模式**：坐标文件 + 可选 `visualization_detection.jpg`；0 目标目录 `{stem}_无目标`。
- **打包**：`build_deploy_wpf.ps1` 校验 YOLO 权重，缺失则失败；`yolo_path` 使用相对路径 `detect_weights/best.pt`。
- **已移除**：`crops/`、`result.json`、`statistics.json` 及 `rebuild_summary_from_stats` 补统计脚本。

## 功能页

钻石检测分类 · 缺陷检测 · 结果管理 · 误分类修正 · 模型再训练 · 设置  
（开发版/机台版行为对齐原 `defect_detect`。）
