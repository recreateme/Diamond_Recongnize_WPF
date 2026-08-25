# 钻石缺陷检测系统 — WPF 操作手册

面向 **不熟悉 WPF** 的开发与维护人员：说明本仓库技术栈、目录模块、日常开发/运行/维护方式。  
实施节奏与阶段划分见同目录 [`WPF重构实施计划.md`](WPF重构实施计划.md)。

| 项 | 内容 |
|----|------|
| 项目根目录 | `D:\Develop\diamond_detect_wpf` |
| 源参考仓 | `D:\Develop\defect_detect`（PyQt5 原版，算法同源） |
| 文档版本 | v1.1 |
| 日期 | 2026-08-25 |

---

## 1. 五分钟看懂本项目

本系统做三件事：

1. **分类**：应用对外三类（棱边朝上 / 点朝上 / 面朝上）。模型权重仍按 5 类训练与导出；推理排除「局部破损」「断钻」后取有效类最高分。
2. **大图检测**：对约 5120×5120 图做 SAHI 切片 + YOLO 检出，再对裁剪块分类。
3. **主动学习**：机台误分类归档 → 开发机微调 → 更新模型回机台。

**架构一句话**：界面用 **WPF（C#）**；训练与推理仍用 **Python 原模块**（`python_core/`），二者经 Bridge 调用。

```
用户操作 → WPF 窗口 (Views/ViewModels)
              ↓
         Bridge（调用 Python）
              ↓
    python_core（推理 / SAHI / 训练）
              ↓
         checkpoints / detect_weights
```

---

## 2. WPF 是什么？（给新手）

| 概念 | 说明 |
|------|------|
| **WPF** | Windows Presentation Foundation，微软桌面 UI 框架，用 **XAML** 描述界面、用 **C#** 写逻辑。 |
| **XAML** | 类似 HTML 的界面标记：`Button`、`Grid`、`ListView` 等。改布局优先改 `.xaml`。 |
| **Code-behind** | 与 XAML 同名的 `.xaml.cs`，本项目尽量少写业务，只做初始化。 |
| **MVVM** | Model–View–ViewModel：界面（View）绑定 ViewModel 属性/命令，避免在按钮点击里堆业务代码。 |
| **数据绑定** | `{Binding Title}` 把控件与 ViewModel 属性连起来；属性变更会刷新界面。 |
| **Dispatcher** | UI 线程调度器。后台线程算完推理后，必须通过它更新界面（对应原 PyQt 的「别在 QThread 里改控件」）。 |

### 与原 PyQt5 对照

| PyQt5 | WPF |
|-------|-----|
| `.py` + 代码拼布局 / QSS | `.xaml` 布局 + `Themes/*.xaml` 样式 |
| `pyqtSignal` / 槽 | 命令 `ICommand`、属性变更 `INotifyPropertyChanged` |
| `QThread` | `Task` + `CancellationToken` + `IProgress` |
| `QStackedWidget` 多页 | `ContentControl` + 切换当前 View |
| `app.py` 巨型单文件 | 按页面拆成多个 View / ViewModel |

### 建议阅读顺序（第一次接触仓库）

1. 本文第 3–5 节（技术栈与目录）
2. `src/DiamondDetect.Wpf/MainWindow.xaml`（壳与导航）
3. `src/DiamondDetect.Core/`（配置、结果模型、接口）
4. `python_core/README.md`（算法入口）
5. [`WPF重构实施计划.md`](WPF重构实施计划.md) 中当前 Phase

---

## 3. 技术栈

### 3.1 界面与宿主（C# / .NET）

| 组件 | 选型 | 作用 |
|------|------|------|
| 运行时 | **.NET 8**（Windows Desktop） | WPF 宿主 |
| UI | **WPF** | 六大功能页 |
| MVVM | CommunityToolkit.Mvvm | `ObservableObject` / `RelayCommand` |
| DI | Microsoft.Extensions.DependencyInjection | 注册 Session、Bridge |
| IDE | **Visual Studio 2026**（主） | XAML 设计、调试、发布 |
| 辅助 IDE | Cursor / VS Code / PyCharm | Python、文档、脚本 |

### 3.2 算法与数据（Python，必须保留）

| 组件 | 位置 | 作用 |
|------|------|------|
| 开发分类 | `python_core/inference_engine.py` | PyTorch GPU 优先 |
| 机台分类 | `python_core/inference_engine_onnx.py` | ONNX Runtime |
| 公共逻辑 | `python_core/inference_common.py` | softmax、有效类别 argmax、批量结果结构 |
| SAHI 流水线 | `python_core/sahi_detector.py` | 大图检测 + 分类 |
| 训练 | `python_core/train.py` | 训练 / 微调 / 导出 ONNX |
| 尺寸分析 | `python_core/analyze_image_sizes.py` | 推荐 `img_size` |
| 路径 / DLL | `python_core/app_paths.py` | 开发与打包路径、ORT CUDA |
| 环境 | Conda **`cv-yolo`**（与原项目一致） | torch / ultralytics / ORT |

### 3.3 桥接

| 方式 | 用途 |
|------|------|
| **pythonnet（主路径）** | WPF 进程内调用 Python 引擎 |
| 本地 IPC（备选） | pythonnet 在机台不稳定时启用 |

**规则**：softmax、有效类别决策、批量结果结构 **只改** `inference_common.py`，禁止在 C# 复制一份。已废弃逐类 `class_thresholds.json`。

### 3.4 配置与产物

| 文件/目录 | 说明 |
|-----------|------|
| `app_config.json` | 模型路径、GPU、SAHI 参数（字段与原版兼容） |
| `checkpoints/` | `best_model.pt`、`model.onnx`、`class_map.json`、`train_config.json`（**不再需要** `class_thresholds.json`） |
| `detect_weights/best.pt` | YOLO 检测权重（机台） |
| `corrections/` | 误分类归档（主动学习） |
| `sahi_output/` | 大图流水线默认输出 |
| `data/` | 训练数据（按类别子文件夹） |

---

## 4. 仓库目录与模块说明

```
diamond_detect_wpf/
├── WPF操作手册.md              ← 本文
├── WPF重构实施计划.md
├── README.md
├── DiamondDetect.sln           ← Visual Studio 打开此文件
├── src/
│   ├── DiamondDetect.Wpf/      ← 界面：Views、ViewModels、Themes
│   ├── DiamondDetect.Core/     ← 领域模型、配置、接口（不依赖 UI）
│   └── DiamondDetect.Bridge/   ← pythonnet / IPC，调用 python_core
├── python_core/                ← ★ 原算法模块（权威实现）
│   ├── inference_*.py
│   ├── sahi_detector.py
│   ├── train.py
│   ├── app_paths.py
│   ├── scripts/                ← 打包与验收
│   └── …
├── checkpoints/                ← 开发期模型（可从源仓复制）
├── docs/                       ← 契约与补充说明
└── tests/                      ← 后续回归测试
```

### 4.1 `DiamondDetect.Wpf`（界面层）

| 内容 | 说明 |
|------|------|
| `MainWindow` | 侧栏导航 + 内容区，对应原 `MainWindow` |
| `Views/*` | 六页：钻石检测、缺陷检测、结果、修正、再训练、设置 |
| `ViewModels/*` | 每页一个 VM：命令、进度、绑定属性 |
| `Controls/*` | 管理员锁、拖放区、预览等可复用控件 |
| `Themes/` | 颜色、按钮样式（替代原 QSS） |

**改界面**：先改 XAML；改交互逻辑改对应 ViewModel；不要把推理代码写进 View。

### 4.2 `DiamondDetect.Core`（领域层）

| 内容 | 说明 |
|------|------|
| `DetectionResult` | 与 Python 结果 dict 字段对齐 |
| `AppConfig` | 对应 `app_config.json` |
| `IInferenceEngine` / `ISahiPipeline` / `ITrainRunner` | 抽象，便于测试与换 Bridge |
| `AppSession` | 共享引擎、结果列表、管理员解锁状态（原 `AppState`） |
| `DeployMode` | 开发版 / 机台版 |

### 4.3 `DiamondDetect.Bridge`（桥接层）

负责：定位 Python 解释器、设置 DLL 路径、`import` 引擎、把 dict 转成 C# 模型、把进度回调到 UI。

出问题优先查：Conda 环境是否激活路径正确、`app_paths.setup_ort_dll_paths` 是否已调用、模型文件是否存在。

### 4.4 `python_core`（算法层，禁止随意替换）

| 模块 | 何时改 |
|------|--------|
| `inference_common.py` | 置信度、阈值、批量结果不一致时 |
| `inference_engine*.py` | 仅加载方式 / Provider / 预处理后端差异 |
| `sahi_detector.py` | 切片、NMS、可视化、过滤参数语义 |
| `train.py` | 训练策略、导出 ONNX |
| `scripts/build_deploy*.py` | 机台打包流程 |

完整保留清单见实施计划「模块保留确认表」。

### 4.5 六大功能页对照

| 导航 | View | 原 PyQt 页 | 主要干什么 |
|------|------|------------|------------|
| 钻石检测分类 | `DiamondDetectView` | `DiamondDetectPage` | SAHI 大图流水线（已实现） |
| 缺陷检测 | `DetectionView` | `DetectionPage` | 单张 / 批量分类 |
| 结果管理 | `ResultsView` | `ResultsPage` | 筛选、导出、预览 |
| 误分类修正 | `CorrectionView` | `CorrectionPage` | 归档到 `corrections/`（已实现） |
| 模型再训练 | `RetrainView` | `RetrainPage` | 调 `train.py`（机台只读，已实现） |
| 设置 | `SettingsView` | `SettingsPage` | 分类 + 切片参数（默认管理员锁） |

---

## 5. 环境准备

### 5.1 必备软件

1. **Visual Studio 2026**：工作负载勾选「**.NET 桌面开发**」。
2. **.NET 8 SDK**（命令行 `dotnet --list-sdks` 应能看到 8.x）。
3. **Conda 环境 `cv-yolo`**：与原 `defect_detect` 相同依赖（见 `python_core/requirements.txt`）。本机示例路径：`D:\Software\MiniAnaconda\envs\cv-yolo`。
4. **NVIDIA 驱动**（机台 / 开发 GPU 推理）；不必单独装完整 CUDA Toolkit（打包会带 DLL）。

若启动提示找不到 Python，可设置：

```powershell
$env:DIAMOND_PYTHON_HOME = "D:\Software\MiniAnaconda\envs\cv-yolo"
```

### 5.2 首次打开工程

```powershell
# 1. 用 VS 打开
#    D:\Develop\diamond_detect_wpf\DiamondDetect.sln

# 2. 或命令行还原并编译
cd D:\Develop\diamond_detect_wpf
dotnet restore
dotnet build

# 3. 确认 Python 算法侧（另开终端）
conda activate cv-yolo
cd D:\Develop\diamond_detect_wpf\python_core
python -c "import inference_common; print('ok')"
```

### 5.3 模型与配置

开发前请保证仓库根或运行目录具备：

- `checkpoints/`（至少机台需要 `model.onnx` + 元数据 JSON；开发分类还需 `best_model.pt`）
- `app_config.json`（可从源仓复制后改相对路径）
- YOLO：`detect_weights/best.pt` 或配置里的 `yolo_path`

可从源仓复制：

```powershell
Copy-Item D:\Develop\defect_detect\checkpoints D:\Develop\diamond_detect_wpf\checkpoints -Recurse -Force
Copy-Item D:\Develop\defect_detect\app_config.json D:\Develop\diamond_detect_wpf\app_config.json -Force
```

---

## 6. 日常开发操作

### 6.1 推荐工作流

| 你要改的内容 | 用哪个 IDE | 改哪里 |
|--------------|------------|--------|
| 按钮、布局、颜色 | Visual Studio | `*.xaml` / `Themes` |
| 页面逻辑、进度条、命令 | Visual Studio / Cursor | `ViewModels` |
| 调用 Python 的方式 | Cursor | `DiamondDetect.Bridge` |
| 识别效果、阈值 | PyCharm / Cursor | `python_core/inference_common.py` |
| SAHI / YOLO | PyCharm / Cursor | `python_core/sahi_detector.py` |
| 训练 | PyCharm | `python_core/train.py` |

### 6.2 运行开发版 WPF

在 Visual Studio 中将启动项目设为 **`DiamondDetect.Wpf`**，按 F5。  
或：

```powershell
cd D:\Develop\diamond_detect_wpf
dotnet run --project src\DiamondDetect.Wpf
```

开发版：分类优先 PyTorch；可再训练。

**常用快捷键（Phase 7）**

| 快捷键 | 作用 |
|--------|------|
| Ctrl+1…6 | 切换六大功能页 |
| F5 | 刷新结果管理 / 误分类修正列表 |
| Esc | 停止批量检测 / SAHI / 训练 |

高 DPI 已启用 PerMonitorV2；大列表使用行虚拟化 + 缩略图 LRU 缓存；结果页筛选/排序为增量刷新。异常会写入仓库根 `logs/crash_*.txt`（不上传）。

排查卡顿时可在「设置 → 分类配置」勾选 **启用本地诊断日志**（`enable_local_diagnostics`，默认关），事件写入 `logs/diag.jsonl`。

### 6.3 模拟机台版

设置环境变量后启动（与原 `DEFECTS_DEPLOY=1` 对齐）：

```powershell
$env:DEFECTS_DEPLOY = "1"
dotnet run --project src\DiamondDetect.Wpf
```

机台版：ONNX 分类、再训练只读、设置默认锁定、启动自动加载 `checkpoints/model.onnx`。

### 6.4 仅测 Python（不启 UI）

```powershell
conda activate cv-yolo
cd D:\Develop\diamond_detect_wpf\python_core
python train.py --help
python scripts\verify_deploy.py
```

### 6.5 调试原则（必读）

1. **耗时推理/训练不要放在 UI 线程**，否则窗口假死。
2. 后台完成后更新界面用 **`Dispatcher`** 或绑定属性（在 UI 同步上下文）。
3. 改算法后，用同一张图对比原 `defect_detect` 与本 WPF 的 `class` / `confidence`。
4. 提交前确认未把含密钥的本地绝对路径写进将要发给机台的 `app_config.json`。

---

## 7. 应用维护

### 7.1 配置维护（`app_config.json`）

| 字段组 | 示例字段 | 说明 |
|--------|----------|------|
| 分类 | `pt_path`, `onnx_path`, `use_gpu` | 开发用 pt；机台用 onnx |
| 数据目录 | `data_dir`, `corrections_dir` | 训练与修正 |
| SAHI | `yolo_path`, `sahi_slice_size`, `sahi_overlap`, … | 切片与过滤；长宽比等与原版语义一致 |
| 输出 | `sahi_output_dir` | 大图结果目录 |

- 机台配置须用 **相对路径**（相对 exe 目录）。
- 改切片参数一般需 **管理员解锁** 后保存。

### 7.2 只更新模型（无需重装程序）

覆盖机台目录下：

```
checkpoints/model.onnx
checkpoints/model.onnx.data   # 若有
checkpoints/class_map.json
checkpoints/train_config.json
```

（应用不再读取 `class_thresholds.json`；打包清单亦已移除该文件。）

重启应用（启动会自动加载）。更新检测：覆盖 `detect_weights/best.pt`。

### 7.2.1 钻石检测 `summary.csv` 与补统计工具

钻石检测分类批量输出根目录的 `summary.csv` 由 Bridge（`PythonSahiPipeline.WriteSummaryCsv`）写入，表头为：

`图像,汇总钻石数,棱边朝上,点朝上,面朝上,检测耗时(s),分类耗时(s),总耗时(s)`

- 多张图时末行：`批次合计`（汇总钻石数 = 各图钻石数之和；三列 = 各类合计）。
- 仅单张时不写批次合计行。
- 「选择文件」时输入区显示各文件完整路径；「选择文件夹」仍显示文件夹路径。

若误删 `summary.csv`，可将工具放到结果根目录（如 `D:\迅雷下载\ECOA`）双击重建：

```
scripts/rebuild_summary_from_stats.py
scripts/build_rebuild_summary_exe.bat   → 生成 dist_tools/rebuild_summary_from_stats.exe
```

扫描规则：结果根下一层子目录的 `*/statistics.json` → 写出同级 `summary.csv`（已有则备份为 `.bak`）。

### 7.3 主动学习闭环

```
机台检测 → 误分类修正 → corrections/
    → 拷回开发机
    → python train.py --finetune --extra_data_dirs corrections
    → 更新 checkpoints
    → 拷回机台
```

### 7.4 管理员锁定

「设置」「模型再训练」默认只读。解锁密码与原版一致（见 Core/配置中的管理员口令约定）。日常检测不需要进设置。

### 7.5 日志与排障

| 现象 | 排查 |
|------|------|
| 分类引擎未加载 | `checkpoints/model.onnx` 是否存在；`app_config.json` 路径；状态栏报错 |
| ORT / CUDA 错误 | 驱动；是否已调用 DLL 路径设置；可先勾选/回退 CPU |
| SAHI / YOLO 报错 | `yolo_path`；`ultralytics`；设备设为 `auto`/`cpu` |
| WPF 启动但无法 import Python | Conda 路径、Bridge 配置的解释器、`PYTHONPATH` 是否含 `python_core` |
| 界面卡死 | 是否在 UI 线程做了批量推理 |
| 单张与批量结果不一致 | 只应走 `inference_common`；检查是否旁路了公共逻辑 |

### 7.6 打包与机台部署

正式入口：

```powershell
.\scripts\build_deploy_wpf.ps1 -PythonHome "D:\Software\MiniAnaconda\envs\cv-yolo"
# 或瘦包：
.\scripts\build_deploy_wpf.ps1 -SkipPythonRuntime
```

产物在 `dist\缺陷分类系统\`。细节见 [docs/打包部署说明.md](docs/打包部署说明.md) 与 [docs/验收清单.md](docs/验收清单.md)。

原则：

1. 开发机在 `cv-yolo` 中验收模型与依赖。
2. 发布 WPF 自包含宿主 + `python_core` +（可选）`python_runtime`。
3. **整包复制**到机台。
4. 用 `验收_verify.bat` / `DEFECTS_VERIFY=1` 无界面验证加载。

---

## 8. 开发版 vs 机台版速查

| 项 | 开发版 | 机台版 |
|----|--------|--------|
| 触发 | 默认 | `DEFECTS_DEPLOY=1` 或发布配置 |
| 分类引擎 | PyTorch（可 ONNX 回退） | 仅 ONNX Runtime |
| 再训练 | 可用 | 只读 |
| 设置 | 解锁后可改 | 默认锁定；启动自动加载 |
| 窗口标识 | 普通标题 | 带「机台版」类标识 |

---

## 9. 常见命令速查

```powershell
# 编译 WPF
cd D:\Develop\diamond_detect_wpf
dotnet build

# 运行开发版
dotnet run --project src\DiamondDetect.Wpf

# 运行机台模拟
$env:DEFECTS_DEPLOY = "1"
dotnet run --project src\DiamondDetect.Wpf

# Python：训练 / 分析 / 验收
conda activate cv-yolo
cd D:\Develop\diamond_detect_wpf\python_core
python analyze_image_sizes.py --data_dir ..\data
python train.py --data_dir ..\data --img_size 128
python train.py --finetune --extra_data_dirs ..\corrections
python train.py --postprocess_only
python scripts\verify_deploy.py
```

---

## 10. 维护责任边界（避免改错地方）

| 需求 | 应改 |
|------|------|
| 按钮文案、布局、主题色 | `DiamondDetect.Wpf` |
| 结果表格列、导出按钮行为 | 对应 ViewModel + Core DTO |
| Python 调不通 / 进度回调 | `DiamondDetect.Bridge` |
| 识别类别错、阈值怪 | `python_core/inference_common.py`（及训练阈值文件） |
| 大图漏检、框异常 | `python_core/sahi_detector.py` + 设置中 SAHI 参数 |
| 打包体积、机台 DLL | `python_core/scripts/` |

---

## 11. 相关文档

| 文档 | 用途 |
|------|------|
| [`WPF重构实施计划.md`](WPF重构实施计划.md) | 阶段任务、风险、验收矩阵 |
| `python_core/README.md` | Python 模块索引与命令 |
| 源仓 `defect_detect/README.md` | 原业务与训练说明（算法仍适用） |
| 源仓 `打包部署说明.md` | 机台经验（打包脚本演进时对照） |

---

**维护提示**：算法以 `python_core` 为唯一修改点；界面以 WPF 项目为唯一修改点。两边契约（配置字段、结果 dict）变更时，请同步更新本文第 3.4 节与 `docs/CONTRACTS.md`（若已创建）。
