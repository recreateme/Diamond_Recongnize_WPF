# 钻石缺陷图像分类系统 — WPF 优化重构实施计划

| 项 | 内容 |
|----|------|
| 源项目 | `D:\Develop\defect_detect`（PyQt5 + Python ML） |
| 目标项目 | `D:\Develop\diamond_detect_wpf`（WPF UI + 保留全部既有业务模块） |
| 文档版本 | v1.2 |
| 日期 | 2026-08-25 |
| 约束 | **现有业务/算法模块必须全部保留**；UI 层由 PyQt5 迁移为 WPF |
| 进度 | **Phase 0–7 已落地**；2026-08 起应用层改为 **3 类**（排除局部破损/断钻）、取消阈值文件；详见 [docs/CONTRACTS.md](docs/CONTRACTS.md) |

---

## 1. 背景与目标

### 1.1 现状摘要

源项目是一套工业钻石缺陷分类系统（EfficientNet-B0 + 主动学习闭环；训练侧曾为五类）。**现行应用**对外三类，模型可仍输出五类 logits。核心能力包括：

| 能力 | 实现 |
|------|------|
| 数据分析 | `analyze_image_sizes.py` |
| 训练 / 微调 / ONNX 导出（训练可含阈值校准，应用侧忽略） | `train.py` |
| 开发版分类推理（PyTorch GPU 优先） | `inference_engine.py` |
| 机台版分类推理（ONNX Runtime） | `inference_engine_onnx.py` |
| 推理公共逻辑（softmax、有效类 argmax、批量） | `inference_common.py` |
| SAHI 大图切片检测 + 分类流水线 | `sahi_detector.py` |
| 路径 / 冻结进程 / ORT DLL | `app_paths.py` |
| PyQt5 桌面 UI（约 4200 行，已由 WPF 替代） | `app.py` |
| 机台入口 | `app_deploy.py` |
| 打包与验收 | `scripts/*`、`pyinstaller_hooks/*` |

当前 GUI 为 **PyQt5**：侧栏导航 + `QStackedWidget` 六页，长任务走 `QThread`，共享 `AppState`。

### 1.2 重构目标

1. **UI 框架**：PyQt5 → **WPF（.NET）**，获得更佳 Windows 原生体验、可维护的 XAML 布局、更清晰的 MVVM 结构。
2. **业务完整保留**：下列模块在新工程中 **继续作为权威实现**，不允许用“近似重写”替代后丢弃：
   - `train.py`
   - `analyze_image_sizes.py`
   - `inference_common.py`
   - `inference_engine.py`
   - `inference_engine_onnx.py`
   - `sahi_detector.py`
   - `app_paths.py`
   - `app_deploy.py`（语义保留：机台入口 / 验收模式）
   - `scripts/build_deploy.py` 及同目录验收脚本、`pyinstaller_hooks/rthook_ort_dll.py`
   - `checkpoints/` 产物约定、`app_config.json` 配置契约、主动学习目录约定（`data/`、`corrections/`、`sahi_output/`）
3. **功能对等**：六大页面、开发版/机台版双模式、管理员锁定、批量推理、SAHI 流水线、修正归档、再训练、打包部署 **行为与源项目一致**。
4. **结构优化**：拆分 `app.py` 巨型单文件；UI 与算法严格分层；便于后续迭代。

### 1.3 非目标（本期不做）

- 不更换分类模型架构（仍为 EfficientNet-B0 / 现有 checkpoints）。
- 不重写 YOLO/SAHI 算法为纯 C#（本期继续调用既有 Python 模块）。
- 不引入 Web/云端服务化（仍为本地桌面 + 机台离线包）。

---

## 2. 源项目模块盘点（必须保留清单）

### 2.1 算法与运行时（Python，保留并纳入新仓库）

| 模块 | 行数约 | 职责 | 重构后处置 |
|------|--------|------|------------|
| `inference_common.py` | 267 | 元数据、softmax、阈值、`run_batch_predict`、统一结果 dict | **原样保留**，唯一权威 |
| `inference_engine.py` | 322 | 开发版 `InferenceEngine`（`load/predict/predict_batch/predict_batch_images`） | **原样保留** |
| `inference_engine_onnx.py` | 316 | 机台 ONNX `InferenceEngine` + Provider 选择 | **原样保留** |
| `sahi_detector.py` | 1297 | `SahiDetector` / `SahiPipeline`、NMS/过滤/可视化 | **原样保留** |
| `train.py` | 1412 | FocalLoss、数据集、训练管线、ONNX 导出、阈值校准 | **原样保留**（仍由子进程或服务调用） |
| `analyze_image_sizes.py` | 437 | 尺寸统计与推荐分辨率 | **原样保留**（CLI 工具） |
| `app_paths.py` | 198 | 开发/冻结路径、ORT DLL 搜索路径 | **原样保留**；WPF 启动时仍须调用或等价封装 |

### 2.2 UI 相关（将被 WPF 替代，但业务语义保留）

| 源符号（`app.py`） | 业务语义 | WPF 对应 |
|--------------------|----------|----------|
| `MainWindow` + 侧栏导航 | 六页切换 | `MainWindow` + `Frame`/`ContentControl` + 导航 VM |
| `DiamondDetectPage` | SAHI 大图检测分类 | `Views/DiamondDetectView` |
| `DetectionPage` | 单张/批量缺陷分类 | `Views/DetectionView` |
| `ResultsPage` | 结果筛选/导出/预览 | `Views/ResultsView` |
| `CorrectionPage` | 误分类归档到 `corrections/` | `Views/CorrectionView` |
| `RetrainPage` | 调 `train.py` 子进程 | `Views/RetrainView` |
| `SettingsPage` | 分类配置 + 切片推理 Tab | `Views/SettingsView` |
| `AppState` | 引擎、results、config、admin | `AppState` / `IAppSession`（C#） |
| `InferenceWorker` | 批量分类后台 | `IProgress` + `Task` / 宿主回调 |
| `SahiPipelineWorker` | SAHI 后台 | 同上 |
| `TrainWorker` | 训练日志流 | 进程 stdout 重定向到 VM |
| `AdminLockBar` | 管理员解锁 | `AdminLockControl` |
| 拖放区 / 缩略图 / 侧栏预览 | 交互组件 | 等价 UserControl |
| `STYLE` QSS | 视觉主题 | `Themes/*.xaml` ResourceDictionary |
| `_DEFAULT_CFG_*` / `app_config.json` | 配置契约 | **字段名与默认值保持兼容** |
| `DEPLOY_ONNX_ONLY` / `DEFECTS_DEPLOY` | 机台开关 | C# `DeployMode` + 环境变量/编译符号 |
| `ADMIN_PAGE_PASSWORD` | 管理员口令 | 配置或同等常量（迁移时不改变默认行为） |

### 2.3 打包与部署（保留并适配）

| 资产 | 处置 |
|------|------|
| `scripts/build_deploy.py` | 保留核心流程；后期增加/并行 **WPF 发布** 步骤 |
| `scripts/verify_deploy.py` / `verify_frozen_sim.py` | 保留；验收对象扩展为新宿主 |
| `pyinstaller_hooks/rthook_ort_dll.py` | 若仍用 PyInstaller 包 Python 侧，保留 |
| `assets/app.ico` | 复用为 WPF 应用图标 |
| `checkpoints/` 约定 | 不变 |
| `requirements.txt` / `requirements-deploy.txt` | 保留 Python 依赖；去掉 PyQt5，改为“可选兼容/过渡期”说明 |

### 2.4 缺陷类别与结果契约（现行约定）

> 历史目标曾为五类全量展示。自 2026-08 起**应用层**改为三类；模型权重可仍为 5 类输出。细节以 [docs/CONTRACTS.md](docs/CONTRACTS.md) 为准。

| 项 | 约定 |
|----|------|
| 有效类 | `棱边朝上` / `点朝上` / `面朝上` |
| 排除类 | `局部破损` / `断钻`（不参与最终判定与 UI 类别列表） |
| 阈值文件 | 不使用 `class_thresholds.json` |

检测结果 dict 字段须与现 UI/引擎一致：

| 字段 | 含义 |
|------|------|
| `path` | 规范化图像路径（同路径 upsert） |
| `class` / `confidence` | 有效类 argmax 预测及概率 |
| `max_class` / `max_confidence` | 与上相同 |
| `all_scores` | 仅有效三类概率 |
| `true_class` / `flagged` | 修正与待处理标记 |

---

## 3. 目标架构

### 3.1 总体原则

```
┌─────────────────────────────────────────────────────────┐
│  DiamondDetect.Wpf  (.NET 8/9 WPF, MVVM)                 │
│  Views / ViewModels / Services(C#) / Themes               │
└──────────────────────────┬──────────────────────────────┘
                           │  Bridge（进程内 pythonnet 或 本地 RPC）
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Python Core（原模块原样迁入 python_core/）               │
│  inference_* / sahi_detector / train / app_paths / …      │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
              checkpoints / detect_weights / app_config.json
```

**关键决策：算法继续用 Python；WPF 只做展示、交互、线程调度与配置持久化。**  
这与源项目文档结论一致（“Qt 层只负责展示…分类与训练在独立模块”），仅把 Qt 换成 WPF。

### 3.2 桥接方案选型（推荐）

| 方案 | 优点 | 风险 | 建议 |
|------|------|------|------|
| **A. pythonnet 进程内嵌** | 延迟低、调用直接、易映射现有 API | 需管理 Python 运行时与 DLL 路径；调试稍复杂 | **开发版首选** |
| **B. 本地 Python 服务（命名管道 / ZeroMQ / FastAPI localhost）** | 进程隔离好、可单独重启 Python | IPC 序列化成本；部署多一个进程 | **机台备选 / 稳健路径** |
| **C. 分类改写为 ORT C#，检测仍调 Python** | 分类路径纯托管 | 双栈；易与 `inference_common` 漂移；违背“模块全部保留”的精神若弃用引擎 | **不推荐作主路径** |

**实施建议：**

- **Phase 1–3**：采用 **方案 A（pythonnet）**，C# 侧封装 `IInferenceEngine`、`ISahiPipeline`、`ITrainRunner`，内部调用现有 Python 类/函数。
- 若机台打包中 pythonnet 稳定性不足，增加 **方案 B 兜底**：同目录启动 `python_host.py`（薄封装，仍 import 原模块），WPF 经命名管道通信。
- **禁止**在 C# 中复制一份阈值/softmax 逻辑；一律走 `inference_common.py`。

### 3.3 逻辑分层（新仓库）

```
diamond_detect_wpf/
├── README.md
├── WPF重构实施计划.md          ← 本文档
├── docs/                       ← 从源项目迁移并修订的说明文档
├── src/
│   ├── DiamondDetect.Wpf/      ← WPF 主程序（开发 + 机台同一解决方案，条件编译/配置区分）
│   ├── DiamondDetect.Core/     ← C# 领域模型、配置、结果 DTO、接口
│   └── DiamondDetect.Bridge/   ← pythonnet / IPC 适配层
├── python_core/                ← ★ 原 Python 模块原样迁入（保留清单全部在此）
│   ├── inference_common.py
│   ├── inference_engine.py
│   ├── inference_engine_onnx.py
│   ├── sahi_detector.py
│   ├── train.py
│   ├── analyze_image_sizes.py
│   ├── app_paths.py
│   ├── app_deploy_shim.py      ← 机台/验收语义（对应原 app_deploy.py）
│   ├── requirements.txt
│   ├── requirements-deploy.txt
│   ├── scripts/
│   ├── pyinstaller_hooks/
│   └── assets/
├── checkpoints/                ← 开发期可软链或复制策略说明
└── tests/
    ├── Wpf.UiTests/            ← 可选 UI 冒烟
    └── python_core/            ← 对引擎/公共逻辑的回归（可先手工脚本）
```

> 过渡期可继续把 `D:\Develop\defect_detect` 当作只读参考仓；`python_core` 以 **复制 + 版本标记** 方式纳入本仓，避免两边长期双改。

### 3.4 开发版 / 机台版

| 模式 | 触发 | 分类引擎 | 再训练 | 设置 |
|------|------|----------|--------|------|
| 开发版 | 默认 Debug / `DeployMode=Full` | `inference_engine.py` | 可写 | 管理员解锁后可写 |
| 机台版 | `DEFECTS_DEPLOY=1` 或 Release-Deploy / 冻结 | `inference_engine_onnx.py` | 只读 | 默认锁定；启动自动加载 ONNX |

行为对齐原 `DEPLOY_ONNX_ONLY` / `app_deploy.py`。

---

## 4. 技术栈与 IDE 分工

### 4.1 推荐技术栈

| 层 | 技术 | 说明 |
|----|------|------|
| UI | **WPF + .NET 8**（若 VS 2026 默认更高则用其 LTS/.NET 9） | XAML + MVVM |
| MVVM | CommunityToolkit.Mvvm | `ObservableObject` / `RelayCommand` |
| DI | Microsoft.Extensions.DependencyInjection | 注册 Bridge / Session |
| 图像 | WPF `BitmapImage` / `WriteableBitmap`；大图预览可辅 OpenCvSharp（可选） | 对齐原缩略图/侧栏预览 |
| Python 宿主 | 现有 Conda **`cv-yolo`** | 与源项目一致 |
| 桥接 | **pythonnet**（主） | 绑定同一解释器 |
| 打包 | WPF：`dotnet publish` / MSIX 或自包含；Python 侧：沿用/演进 `build_deploy.py` | 见第 8 节 |
| 测试 | xUnit（C#）+ 现有 `verify_*.py` | 功能对等验收 |

### 4.2 本机 IDE 分工建议

| IDE | 用途 |
|-----|------|
| **Visual Studio 2026** | WPF 解决方案主 IDE：XAML 设计器、调试、发布、安装器 |
| **Cursor** | Python Core 改造、Bridge 脚本、文档、跨语言联调辅助 |
| **VS Code** | 轻量编辑 Python / JSON / Markdown；可选 Python 扩展 |
| **PyCharm** | 训练、数据分析、推理单测、Conda 环境管理（可选主力） |

说明：WPF 设计器体验以 Visual Studio 为最佳；算法与训练继续在 PyCharm/Cursor 中维护 `python_core`。

### 4.3 环境前置检查清单

- [ ] 安装 .NET SDK（与 VS 2026 匹配的 8 或 9）
- [ ] Visual Studio 勾选「.NET 桌面开发」工作负载
- [ ] Conda 环境 `cv-yolo` 可运行 `python app.py`（源项目基线）
- [ ] 确认 GPU：`torch.cuda.is_available()` / ORT CUDA Provider
- [ ] 准备 `checkpoints/` 与 YOLO `detect_weights/best.pt`（或配置中的 `yolo_path`）

---

## 5. 功能对等矩阵（验收用）

| 功能 | 源项目 | WPF 目标 | 验收标准 |
|------|--------|----------|----------|
| 钻石检测分类 | `DiamondDetectPage` | 同 | 多图/文件夹；输出裁剪、双可视化、JSON；可打开目录 |
| 缺陷检测·单张 | `DetectionPage` | 同 | 拖放/选图；得分条；阈值提示；upsert 结果 |
| 缺陷检测·批量 | `InferenceWorker` | 同 | 进度条；可停止；batch 推理 |
| 结果管理 | `ResultsPage` | 同 | 筛选、CSV、按类导出、缩略图、真值下拉 |
| 误分类修正 | `CorrectionPage` | 同 | 写入 `corrections/<类>/`；导航角标计数 |
| 模型再训练 | `RetrainPage` | 同 | 开发版可跑 `train.py`；机台只读；日志实时 |
| 设置·分类 | Tab1 | 同 | pt/onnx 路径、GPU；机台无 pt |
| 设置·切片 | Tab2 | 同 | YOLO、slice、overlap、conf、batch、padding、IoS、面积比、长宽比、边缘过滤 |
| 启动自动加载 | `_auto_load_model` | 同 | 机台启动即 ONNX 成功提示 |
| 管理员锁 | `AdminLockBar` | 同 | 密码错误拒绝；解锁后可编辑；可再锁定 |
| 配置持久化 | `app_config.json` | 同 | 字段兼容；机台用相对路径 |
| 打包部署 | `build_deploy.py` | 演进 | 机台可离线跑通 GPU 分类 + SAHI |
| 主动学习闭环 | corrections → finetune | 同 | 路径与命令兼容 |

---

## 6. 分阶段实施计划

### Phase 0 — 基线冻结与仓库初始化（约 0.5–1 天）

**产出**

- 本仓目录结构创建；`python_core/` 从 `defect_detect` 复制保留模块。
- 记录源项目可运行基线（开发 GUI、机台 `app_deploy.py`、一次 SAHI 样例）。
- 制定 `app_config.json` / 结果 dict / 环境变量契约文档（可放 `docs/CONTRACTS.md`）。

**任务**

1. 复制必须保留的 Python 文件与 `scripts/`、`pyinstaller_hooks/`、`assets/`。
2. 从 `app.py` **抽离**纯业务函数到 `python_core/ui_support.py`（可选但推荐）：`_upsert_result`、`export_classified_images`、`_save_correction_to_disk` 等，使 WPF 与旧 Qt 都能调用同一逻辑（过渡期）。
3. 解决方案 `DiamondDetect.sln` 建好三个 C# 项目骨架。

**退出条件**：`python_core` 下 `train.py --help`、引擎 load 冒烟通过；WPF 空窗体能启动。

---

### Phase 1 — Bridge 与会话层（约 3–5 天）

**目标**：C# 能稳定调用 Python 推理，不依赖任何完整页面。

**任务**

1. 实现 `PythonRuntimeHost`：定位 `cv-yolo` 解释器、`PYTHONHOME`/`PATH`、调用 `app_paths.setup_ort_dll_paths()`。
2. 封装：
   - `IInferenceEngine` → `InferenceEngine.load/predict/predict_batch/predict_batch_images`
   - 结果映射：Python dict → `DetectionResult` record（字段 1:1）
3. 封装 `ISahiPipeline` → `SahiPipeline.process_image(s)`，进度回调用 `IProgress<T>`。
4. 封装 `ITrainRunner` → `subprocess` 等价（`Process` 读 stdout）。
5. 单元/集成测试：单张图分类、小批量、错误路径（模型缺失）。

**退出条件**：控制台或临时按钮即可完成「加载 ONNX → 推理一张图 → 打印类别置信度」。

---

### Phase 2 — 壳与导航 + 设置 / 自动加载（约 3–4 天）

**任务**

1. `MainWindow`：侧栏 + 内容区；六页导航索引对齐 `NAV_*`。
2. `AppSession`：config、engine、results、`AdminUnlocked`。
3. `SettingsView` 双 Tab；读写 `app_config.json`（迁移逻辑对齐 `_migrate_cfg`）。
4. 启动 `_AutoLoadModel`（机台默认 onnx）。
5. 主题 ResourceDictionary（先功能后美观，避免阻塞）。

**退出条件**：开发/机台两种模式启动；状态栏显示引擎类型与 GPU/CPU。

---

### Phase 3 — 缺陷检测 + 结果管理（约 4–6 天）

**任务**

1. `DetectionView`：单张（拖放、得分条、阈值提示）+ 批量（进度、停止）。
2. 后台 `Task.Run` + `Dispatcher` 更新 UI（对应原 QThread 规则）。
3. `ResultsView`：表格、筛选、CSV、按类导出、侧栏预览（悬停/固定）。
4. 路径规范化与 upsert 与源一致。

**退出条件**：与源项目对同一文件夹批量结果一致（类别与 confidence；允许浮点误差）。

---

### Phase 4 — 钻石检测分类（SAHI）（约 4–6 天）

**任务**

1. `DiamondDetectView`：输入多选/文件夹、输出目录、开始/停止。
2. 进度：`progress` / `image_done` / `finished_all` / `error` 映射到 VM。
3. 完成后打开目录；展示统计摘要（对齐原 JSON/CSV）。
4. 设备 `auto` 回退行为保持（含新型号 GPU 回退 CPU）。

**退出条件**：对同一张 5120×5120 样例，输出文件集合与主要计数字段与源项目一致。

---

### Phase 5 — 误分类修正 + 再训练（约 2–4 天）

**任务**

1. `CorrectionView`：标记、归档、角标计数。
2. `RetrainView`：参数表单、日志、`model_updated` 后热加载。
3. 机台模式强制只读。
4. 管理员锁贯通 Retrain/Settings。

**退出条件**：完整主动学习闭环在开发机跑通：检测 → 修正 → finetune → 更新 checkpoints → 再推理。

---

### Phase 6 — 打包、机台部署与文档（约 3–5 天）

**任务**

1. 设计最终交付形态（二选一或组合，见第 8 节）。
2. 改造/扩展 `scripts/build_deploy.py`（或新增 `build_deploy_wpf.py`）：收集 CUDA DLL、checkpoints、YOLO、写机台 `app_config.json`。
3. 保留 `DEFECTS_VERIFY` 语义的无 GUI 验收。
4. 迁移并改写文档：`README`、`打包部署说明`、新增 `WPF应用开发说明`。
5. 机台实机验收清单执行。

**退出条件**：机台整包复制后可运行；GPU 分类 + SAHI；仅更新 `checkpoints/` 可热换模型。

---

### Phase 7 — 优化与清理（持续 / 约 2–3 天）✅

- 性能：大表虚拟化（`VirtualizedDataGridStyle` / ListBox Recycling）、`ThumbnailCache` 解码限幅 + LRU、缩略图后台分批加载、**结果表按 StoreIndex 增量同步**（筛选/排序复用行与缩略图）。
- 高 DPI：`PerMonitorV2`（app.manifest + ApplicationHighDpiMode）、`UseLayoutRounding` / `SnapsToDevicePixels`。
- UX：Ctrl+1…6 导航、F5 刷新结果/修正、Esc 停止长任务；`UserMessage` 友好错误文案；顶部忙碌条联动。
- 可维护性：删除过渡 `PlaceholderPage`；`BridgeApi.Version` 接口版本标记；本地 `logs/crash_*.txt` 崩溃日志（默认不上传）。
- 可选诊断：`enable_local_diagnostics`（默认关）→ `logs/diag.jsonl`，设置页可开关。

---

## 7. 从 `app.py` 迁移的详细映射

### 7.1 页面与 ViewModel

| 原类 | View | ViewModel | 主要依赖 |
|------|------|-----------|----------|
| `MainWindow` | `MainWindow.xaml` | `MainViewModel` | `AppSession` |
| `DiamondDetectPage` | `DiamondDetectView` | `DiamondDetectViewModel` | `ISahiPipeline` |
| `DetectionPage` | `DetectionView` | `DetectionViewModel` | `IInferenceEngine` |
| `ResultsPage` | `ResultsView` | `ResultsViewModel` | `AppSession.Results` |
| `CorrectionPage` | `CorrectionView` | `CorrectionViewModel` | 归档服务 |
| `RetrainPage` | `RetrainView` | `RetrainViewModel` | `ITrainRunner` |
| `SettingsPage` | `SettingsView` | `SettingsViewModel` | 配置 IO |

### 7.2 线程模型对照

| PyQt5 | WPF |
|-------|-----|
| `QThread.run` + `pyqtSignal` | `Task` / `Channel` + `IProgress`；UI 更新用 `Dispatcher.InvokeAsync` |
| 禁止子线程改控件 | 同：仅 VM 属性变更经同步上下文 |
| `stop_flag` | `CancellationToken` |

### 7.3 建议优先抽到 `python_core` 的纯函数

便于 Bridge 与回归测试（无需 GUI）：

- `_normalize_image_path` / `_upsert_result` / `_ensure_result_meta`
- `export_classified_images` / CSV 导出逻辑
- `_save_correction_to_disk` / `_archive_results_at_indices`
- 配置 load/save/migrate（可做成 `config_io.py`）

`app.py` 中的控件类不再迁移，只作行为参考。

---

## 8. 打包与部署策略

### 8.1 推荐交付形态（机台）

**方案：自包含 WPF 宿主 + 旁路 Python 运行时目录（或嵌入 conda-pack）**

```
缺陷分类系统/
├── DiamondDetect.exe          ← WPF
├── app_config.json
├── checkpoints/
├── detect_weights/
├── python_runtime/            ← 精简嵌入式 Python + 依赖
│   └── …（含 onnxruntime、torch/ultralytics 按需）
└── python_core/               ← 业务脚本
```

体积仍可能接近现网 ~6GB（torch + CUDA），与现 PyInstaller 方案同级属预期。

### 8.2 与现脚本关系

| 步骤 | 动作 |
|------|------|
| 校验 ORT / 模型 | 保留 `verify_deploy.py` |
| 收集 CUDA DLL | 复用 `build_deploy.py` 中 staging 逻辑 |
| 写机台配置 | 相对路径、`sahi_max_aspect_ratio` 等默认值对齐 |
| 生成 exe | 改为 `dotnet publish -c Release`（替代仅打包 `app_deploy.py`） |
| 无 GUI 验收 | WPF 支持 `--verify` 或环境变量 `DEFECTS_VERIFY=1` 调 Bridge 后退出 |

过渡期允许：**旧 PyQt exe 与新 WPF 包并行**，以同一套 `checkpoints` 验收一致性。

### 8.3 机台运维不变项

- 只更新模型：覆盖 `checkpoints/` 后重启。
- 更新 YOLO：覆盖 `detect_weights/best.pt`。
- `corrections/` 拷回开发机微调。
- GPU 优先、失败回退 CPU。

---

## 9. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| pythonnet 与 conda/CUDA DLL 路径冲突 | 启动失败 / ORT 找不到 cuDNN | 启动最早调用 `setup_ort_dll_paths`；参考 `rthook_ort_dll.py`；准备 IPC 备选 |
| 结果浮点/预处理不一致 | 验收失败 | 禁止 C# 重写 softmax/阈值；共用 `inference_common`；固定测试集对比 |
| `app.py` 隐性逻辑遗漏 | 功能缺口 | 按第 5 节矩阵逐项勾选；对照 `QT应用开发说明.md` |
| 打包体积与路径 | 机台部署失败 | 沿用现 staging/补丁经验；文档化「必须整包复制」 |
| 双仓库漂移 | 修 bug 只改一边 | Phase 0 后以 `diamond_detect_wpf/python_core` 为唯一修改点 |
| 高 DPI / 大图内存 | UI 卡顿 | 缩略图解码限幅；列表虚拟化 |

---

## 10. 测试与验收计划

### 10.1 自动化 / 脚本

1. `python_core`：`verify_deploy.py` 等价（ORT + model.onnx）。
2. Bridge 集成测试：固定样例图 → 期望类别快照。
3. WPF：`--verify` 退出码。

### 10.2 手工场景（必须）

1. 开发版：加载 `.pt`，单张 + 批量 50 张，结果管理导出。
2. 机台模式：仅 ONNX，自动加载，设置默认锁定。
3. SAHI：至少 1 张大图，检查输出文件与可视化。
4. 修正 → 再训练（开发）→ 模型热更新。
5. 管理员错误密码 / 正确密码 / 再锁定。
6. 无 GPU 或强制 CPU：分类仍可用。
7. 同路径重复检测：upsert 不增行。

### 10.3 对比验收

对同一输入，源 `app.py` 与新 WPF：

- 预测 `class` 一致率 100%（同模型、同配置）。
- `confidence` 绝对误差 &lt; 1e-5（或文档约定）。
- SAHI 检出数量允许小幅差异仅当 NMS 随机性存在；同 seed/同参数下应对齐。

---

## 11. 人员与工期粗估

| 阶段 | 工期（人天） |
|------|----------------|
| Phase 0 基线 | 0.5–1 |
| Phase 1 Bridge | 3–5 |
| Phase 2 壳+设置 | 3–4 |
| Phase 3 检测+结果 | 4–6 |
| Phase 4 SAHI | 4–6 |
| Phase 5 修正+训练 | 2–4 |
| Phase 6 打包文档 | 3–5 |
| Phase 7 优化 | 2–3 |
| **合计** | **约 22–34 人天** |

单人全职约 **5–7 周**；若 Bridge 风险高，预留 1 周缓冲给 IPC 备选方案。

---

## 12. 实施顺序建议（落地时按此执行）

1. **不要先画全套精美 UI** — 先打通 Bridge + 自动加载模型。
2. **不要改 `inference_common` / 阈值语义** — UI 迁移期间算法冻结，除非修 bug。
3. **配置与结果契约先行文档化** — 减少返工。
4. **每完成一页就对照源项目点验** — 避免期末大爆炸验收。
5. **打包放到功能对等之后** — 但 Phase 1 就要考虑 DLL 路径，以免末期推翻架构。

---

## 13. 近期立即行动项（Kickoff）

1. 在 `D:\Develop\diamond_detect_wpf` 创建解决方案与 `python_core/`，复制保留模块。
2. 用 Visual Studio 2026 新建 WPF 应用（.NET 8/9），验证本机 SDK。
3. 在 Cursor/PyCharm 中确认 `cv-yolo` 对 `python_core.inference_engine_onnx` 的 load 冒烟。
4. Spike（1–2 天）：pythonnet 调用 `InferenceEngine.predict` 一张图；失败则 Spike IPC。
5. Spike 通过后按 Phase 2→5 推进；同步维护本文档进度勾选。

---

## 14. 附录 A — 源项目导航与页面索引

| 索引 | 常量 | 页面 |
|------|------|------|
| 0 | `NAV_DIAMOND` | 钻石检测分类 |
| 1 | `NAV_DETECT` | 缺陷检测 |
| 2 | `NAV_RESULTS` | 结果管理 |
| 3 | `NAV_CORRECT` | 误分类修正 |
| 4 | `NAV_RETRAIN` | 模型再训练 |
| 5 | `NAV_SETTINGS` | 设置 |

## 附录 B — `app_config.json` 字段（须兼容）

`pt_path`, `onnx_path`, `data_dir`, `corrections_dir`, `use_gpu`, `yolo_path`, `sahi_device`, `sahi_slice_size`, `sahi_overlap`, `sahi_det_conf`, `sahi_batch_size`, `sahi_crop_padding`, `sahi_output_dir`, `sahi_ios_thresh`, `sahi_min_area_ratio`, `sahi_max_aspect_ratio`, `sahi_edge_filter`, `sahi_edge_margin_px`

## 附录 C — 参考文档（源仓）

- `D:\Develop\defect_detect\README.md`
- `D:\Develop\defect_detect\QT应用开发说明.md`（交互与线程模式参考；实现改为 WPF）
- `D:\Develop\defect_detect\打包部署说明.md`

## 附录 D — 模块保留确认表（签字用）

| 模块 | 保留 | 迁入路径 | 确认 |
|------|------|----------|------|
| `inference_common.py` | 是 | `python_core/` | ☐ |
| `inference_engine.py` | 是 | `python_core/` | ☐ |
| `inference_engine_onnx.py` | 是 | `python_core/` | ☐ |
| `sahi_detector.py` | 是 | `python_core/` | ☐ |
| `train.py` | 是 | `python_core/` | ☐ |
| `analyze_image_sizes.py` | 是 | `python_core/` | ☐ |
| `app_paths.py` | 是 | `python_core/` | ☐ |
| `app_deploy.py` 语义 | 是 | `python_core/app_deploy_shim.py` 等 | ☐ |
| `scripts/*` | 是 | `python_core/scripts/` | ☐ |
| `pyinstaller_hooks/*` | 是 | `python_core/pyinstaller_hooks/` | ☐ |
| `assets/app.ico` | 是 | `python_core/assets/` 或 `src/.../Assets` | ☐ |
| `checkpoints` 约定 | 是 | 根目录/部署目录 | ☐ |
| PyQt5 `app.py` UI | 否（由 WPF 替代） | 行为保留在 Views/VMs | ☐ |

---

**文档结束。** 确认本计划后，可从 **第 13 节 Kickoff** 与 **Phase 0** 开始实施。
