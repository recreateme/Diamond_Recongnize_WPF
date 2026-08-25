# 钻石缺陷图像分类系统（WPF 重构版）

工业钻石缺陷五分类：**WPF 桌面界面** + 保留原 Python 算法（`python_core/`）。

| 文档 | 说明 |
|------|------|
| [WPF操作手册.md](WPF操作手册.md) | 技术栈、模块、开发与维护（**新手先读**） |
| [docs/WPF应用开发说明.md](docs/WPF应用开发说明.md) | 页面/MVVM/Bridge 开发指南 |
| [docs/打包部署说明.md](docs/打包部署说明.md) | 机台打包与部署 |
| [docs/验收清单.md](docs/验收清单.md) | 功能验收勾选表 |
| [docs/CONTRACTS.md](docs/CONTRACTS.md) | 配置与结果契约 |
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
| `python_core/` | 原算法权威实现（勿用 C# 重写阈值逻辑） |
| `scripts/` | `publish_wpf` / `build_deploy_wpf` / `verify_wpf` |

## 功能页

钻石检测分类 · 缺陷检测 · 结果管理 · 误分类修正 · 模型再训练 · 设置  
（开发版/机台版行为对齐原 `defect_detect`。）
