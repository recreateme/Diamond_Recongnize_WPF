using System.IO;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using DiamondDetect.Core;
using DiamondDetect.Core.Abstractions;
using DiamondDetect.Core.Services;
using DiamondDetect.Wpf.Services;

namespace DiamondDetect.Wpf.ViewModels;

public partial class MainViewModel : ObservableObject
{
    public const int NavDiamond = 0;
    public const int NavDetect = 1;
    public const int NavResults = 2;
    public const int NavCorrect = 3;
    public const int NavRetrain = 4;
    public const int NavSettings = 5;

    private readonly AppSession _session;
    private readonly IInferenceEngine _engine;
    private readonly IPythonRuntimeHost _pythonHost;
    private readonly IConfigService _configService;

    private Action? _refreshResults;
    private Action? _refreshCorrection;
    private Action? _stopDetection;
    private Action? _stopDiamond;
    private Action? _stopTrain;

    public MainViewModel(
        AppSession session,
        IInferenceEngine engine,
        IPythonRuntimeHost pythonHost,
        IConfigService configService)
    {
        _session = session;
        _engine = engine;
        _pythonHost = pythonHost;
        _configService = configService;
        WindowTitle = $"钻石缺陷图像分类系统 {AppSession.AppVersion}{session.TitleSuffix}";
        StatusText = "正在初始化…";
    }

    public AppSession Session => _session;

    public void AttachPageActions(
        Action refreshResults,
        Action refreshCorrection,
        Action stopDetection,
        Action stopDiamond,
        Action stopTrain)
    {
        _refreshResults = refreshResults;
        _refreshCorrection = refreshCorrection;
        _stopDetection = stopDetection;
        _stopDiamond = stopDiamond;
        _stopTrain = stopTrain;
    }

    [ObservableProperty]
    private string windowTitle = "";

    [ObservableProperty]
    private string statusText = "";

    [ObservableProperty]
    private int selectedNavIndex;

    [ObservableProperty]
    private string engineInfo = "引擎未加载";

    [ObservableProperty]
    private bool isBusy;

    public async Task InitializeAsync()
    {
        using var _ = LocalDiagnostics.Measure("engine.init");
        try
        {
            IsBusy = true;
            StatusText = "初始化 Python 运行时…";
            _pythonHost.Initialize();

            StatusText = "自动加载分类模型…";
            var cfg = _configService.Current;
            LocalDiagnostics.Configure(_session.AppRoot, cfg.EnableLocalDiagnostics);
            var root = _session.AppRoot;
            var pt = ResolvePath(root, cfg.PtPath);
            var onnx = PackLayout.ResolveOnnx(root, cfg.OnnxPath);
            if (string.IsNullOrEmpty(onnx))
                onnx = ResolvePath(root, cfg.OnnxPath);

            string msg;
            if (_session.IsDeploy)
            {
                msg = await Task.Run(() => _engine.Load(null, onnx, cfg.UseGpu));
            }
            else
            {
                msg = await Task.Run(() => _engine.Load(
                    File.Exists(pt) ? pt : null,
                    File.Exists(onnx) ? onnx : null,
                    cfg.UseGpu));
            }

            _session.Engine = _engine;
            var yolo = PackLayout.ResolveYolo(root, cfg.YoloPath);
            var yoloOk = File.Exists(yolo);
            EngineInfo = $"{_engine.Backend} / {_engine.Device} · 类别 {_engine.Classes.Count}"
                         + (yoloOk ? " · YOLO 已就绪" : " · 未找到 YOLO");
            StatusText = msg + (yoloOk ? "  ·  钻石检测权重已就绪" : "  ·  缺少 detect_weights/best.pt");
            _session.StatusText = StatusText;
            WriteStartupLog(root, msg, onnx, yolo, yoloOk);
            LocalDiagnostics.Event("engine.loaded", $"{_engine.Backend}/{_engine.Device};yolo={yoloOk}");
        }
        catch (Exception ex)
        {
            EngineInfo = "引擎未加载";
            StatusText = "启动加载失败: " + UserMessage.Format(ex).Split('\n')[0];
            _session.StatusText = StatusText;
            UserMessage.Error("启动失败", ex);
        }
        finally
        {
            IsBusy = false;
        }
    }

    public void NavigateTo(int index)
    {
        SelectedNavIndex = index;
        LocalDiagnostics.Event("nav", $"page={index}");
    }

    [RelayCommand]
    private void Navigate(object? param)
    {
        if (param is null) return;
        if (int.TryParse(param.ToString(), out var idx))
            NavigateTo(idx);
    }

    [RelayCommand]
    private void RefreshCurrent()
    {
        switch (SelectedNavIndex)
        {
            case NavResults:
                _refreshResults?.Invoke();
                StatusText = "已刷新结果列表";
                break;
            case NavCorrect:
                _refreshCorrection?.Invoke();
                StatusText = "已刷新修正列表";
                break;
            default:
                StatusText = "当前页无列表可刷新（结果/修正页可用 F5）";
                break;
        }
    }

    [RelayCommand]
    private void StopBusy()
    {
        _stopDetection?.Invoke();
        _stopDiamond?.Invoke();
        _stopTrain?.Invoke();
        StatusText = "已发送停止请求";
    }

    [ObservableProperty]
    private string correctionNavText = "误分类修正";

    public void UpdateCorrectionBadge(int pending)
    {
        CorrectionNavText = pending > 0 ? $"误分类修正 ({pending})" : "误分类修正";
    }

    private static string ResolvePath(string root, string? relative)
    {
        if (string.IsNullOrWhiteSpace(relative))
            return "";
        if (Path.IsPathRooted(relative))
            return relative;
        return Path.GetFullPath(Path.Combine(root, relative));
    }

    private void WriteStartupLog(string root, string engineMsg, string onnx, string yolo, bool yoloOk)
    {
        try
        {
            var dir = Path.Combine(root, "logs");
            Directory.CreateDirectory(dir);
            File.WriteAllText(Path.Combine(dir, "startup.txt"),
                $"time={DateTime.Now:O}\n" +
                $"deploy={_session.IsDeploy}\n" +
                $"appRoot={root}\n" +
                $"pythonHome={_pythonHost.PythonHome}\n" +
                $"pythonDll={_pythonHost.PythonDll}\n" +
                $"onnx={onnx} exists={File.Exists(onnx)}\n" +
                $"yolo={yolo} exists={yoloOk}\n" +
                $"engine={engineMsg}\n");
        }
        catch { /* ignore */ }
    }
}
