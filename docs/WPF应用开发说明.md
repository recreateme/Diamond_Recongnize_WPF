# WPF 应用开发说明

面向在本仓库改界面 / Bridge 的开发者。  
产品功能与算法见 [应用功能与算法说明.md](应用功能与算法说明.md)；算法目录见 `python_core/README.md`；契约见 [CONTRACTS.md](CONTRACTS.md)。

---

## 1. 解决方案结构

```
DiamondDetect.sln
├── DiamondDetect.Wpf      # Views / ViewModels / Themes（UI）
├── DiamondDetect.Core     # AppConfig、DetectionResult、ResultStore、接口
└── DiamondDetect.Bridge   # pythonnet、SAHI、TrainRunner
```

原则：**UI 不写 softmax / 有效类决策；算法只改 `python_core`（尤其 `inference_common.py`）。** 应用对外 3 类，见 [CONTRACTS.md](CONTRACTS.md)。

---

## 2. 页面与导航

| 索引 | 页面 | View / ViewModel |
|------|------|------------------|
| 0 | 钻石检测分类 | `DiamondDetectView` |
| 1 | 均匀度分析 | `UniformityView` |
| 2 | 缺陷检测 | `DetectionView` |
| 3 | 结果管理 | `ResultsView` |
| 4 | 误分类修正 | `CorrectionView` |
| 5 | 模型再训练 | `RetrainView` |
| 6 | 设置 | `SettingsPage` |

跨页共享：`AppSession` + `ResultStore`（DI 单例）。

快捷键：`Ctrl+1…7` 导航，`F5` 刷新结果/修正，`Esc` 停止长任务。

均匀度算法：`python_core/diamond_uniformity.py`，Bridge：`IUniformityAnalyzer` / `PythonUniformityAnalyzer`。

---

## 3. MVVM 与线程

- 绑定属性 / `RelayCommand`（CommunityToolkit.Mvvm）。
- 长任务：`Task.Run` + `IProgress` + `CancellationToken`；顶栏 `IsBusy` 与进度条联动。
- **禁止**在后台线程直接改 UI 控件；通过属性变更或 `Dispatcher`。
- 缩略图请走 `Services/ThumbnailCache`（解码限幅 + LRU），勿全分辨率 `BitmapImage`。
- 结果列表刷新走增量同步（`ResultsViewModel.ApplyRowsIncremental`），避免筛选时整表重建。
- 弹窗文案优先 `UserMessage`（附常见排查提示）。
- Bridge 契约版本：`DiamondDetect.Core.BridgeApi.Version`。
- 本地诊断：`LocalDiagnostics` + 配置项 `enable_local_diagnostics`（默认关）。

对照原 PyQt：`QThread` + `pyqtSignal` → 上述模式。

---

## 4. Bridge 要点

| 类型 | 作用 |
|------|------|
| `PythonRuntimeHost` | 定位 Conda、`PYTHONHOME`、初始化 pythonnet、`sys.path` + `setup_ort_dll_paths` |
| `PythonInferenceEngine` | `inference_engine` / `inference_engine_onnx` |
| `PythonSahiPipeline` | `sahi_detector` 逐图处理；`should_stop` 协作取消；完整模式写 `summary.csv`（`SahiSummaryCsv`） |
| `CancellationBridge` | 将 `CancellationToken` 暴露为 Python `should_stop()` |
| `ProcessTrainRunner` | 子进程 `train.py` |

环境变量：

- `DIAMOND_PYTHON_HOME` / `DIAMOND_PYTHON_DLL`
- `DEFECTS_DEPLOY=1` 机台
- `DEFECTS_VERIFY=1` 或 `--verify` 无 GUI 验收

分类决策在 Python `inference_common`（有效 3 类 argmax）；UI / 修正页类别列表来自 `IInferenceEngine.Classes`。
---

## 5. 新增功能页步骤

1. 在 `Views/` 建 UserControl，在 `ViewModels/` 建 VM。
2. `App.xaml.cs` 注册单例。
3. `MainWindow.xaml` 增加 TabItem，code-behind 设 `DataContext`。
4. 侧栏 RadioButton `CommandParameter` 对齐索引。

---

## 6. 调试建议

```powershell
$env:DIAMOND_PYTHON_HOME = "D:\Software\MiniAnaconda\envs\cv-yolo"
dotnet run --project src\DiamondDetect.Wpf
```

- 先看状态栏是否「模型加载成功」。
- Bridge 异常看 Output / 弹窗；ORT 问题优先查 DLL 路径。
- 未处理异常会落盘到 `logs/crash_*.txt`。
- 改算法后用同一张图对比 `defect_detect` 旧版类别与置信度。

---

## 7. 发布

见 [打包部署说明.md](打包部署说明.md)。日常只改 UI：`scripts\publish_wpf.ps1`。
