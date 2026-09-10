using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Windows;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using DiamondDetect.Core;
using DiamondDetect.Core.Abstractions;
using DiamondDetect.Core.Services;
using DiamondDetect.Wpf.Services;
using Microsoft.Win32;

namespace DiamondDetect.Wpf.ViewModels;

public partial class SahiResultRow : ObservableObject
{
    public string Image { get; init; } = "";
    public string DiamondsText { get; init; } = "0";
    public string Distribution { get; init; } = "";
    public string TimeText { get; init; } = "";
    public string OutputDir { get; init; } = "";
}

public partial class DiamondDetectViewModel : ObservableObject
{
    private readonly AppSession _session;
    private readonly IInferenceEngine _engine;
    private readonly ISahiPipeline _sahi;
    private readonly MainViewModel _main;
    private readonly UniformityViewModel _uniformity;
    private CancellationTokenSource? _cts;
    private List<string> _imgPaths = new();
    private string _inputFolder = "";

    public DiamondDetectViewModel(
        AppSession session,
        IInferenceEngine engine,
        ISahiPipeline sahi,
        MainViewModel main,
        UniformityViewModel uniformity)
    {
        _session = session;
        _engine = engine;
        _sahi = sahi;
        _main = main;
        _uniformity = uniformity;
        OutputDir = ResolveDefaultOutput();
    }

    public ObservableCollection<SahiResultRow> Rows { get; } = new();
    public IReadOnlyList<string> InterpolationOptions { get; } =
        new[] { "area", "linear", "cubic", "nearest" };

    [ObservableProperty] private string pageHintText =
        "大图 SAHI 切片检测 + 缺陷分类。可选「仅检测定位」输出坐标 JSON/CSV；「输出选项」可保存可视化图。";
    [ObservableProperty] private bool detectOnly;
    [ObservableProperty] private bool saveVisualization;
    [ObservableProperty] private bool runUniformityAfter;
    [ObservableProperty] private bool downsampleEnabled;
    [ObservableProperty] private int downsampleMaxSide = 2560;
    [ObservableProperty] private string selectedInterpolation = "area";
    [ObservableProperty] private string inputPathText = "未选择图像";
    [ObservableProperty] private string imageCountText = "";
    [ObservableProperty] private string outputDir = "";
    [ObservableProperty] private bool isRunning;
    [ObservableProperty] private bool isStopping;
    [ObservableProperty] private bool progressVisible;
    [ObservableProperty] private int progressValue;
    [ObservableProperty] private int progressMaximum = 1;
    [ObservableProperty] private string stageText = "";
    [ObservableProperty] private bool stageVisible;
    [ObservableProperty] private string totalText = "合计：0 张图 · 0 颗钻石";

    public bool CanStop => IsRunning && !IsStopping;

    partial void OnIsRunningChanged(bool value) => OnPropertyChanged(nameof(CanStop));
    partial void OnIsStoppingChanged(bool value) => OnPropertyChanged(nameof(CanStop));

    [RelayCommand]
    private void BrowseFiles()
    {
        var dlg = new OpenFileDialog
        {
            Multiselect = true,
            Filter = "图像文件|*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff;*.webp|所有文件|*.*",
        };
        if (dlg.ShowDialog() != true || dlg.FileNames.Length == 0) return;
        _imgPaths = dlg.FileNames.Where(ImageFormats.IsImage).ToList();
        _inputFolder = "";
        UpdateFileDisplay();
    }

    [RelayCommand]
    private void BrowseFolder()
    {
        var dlg = new OpenFolderDialog { Title = "选择图像文件夹" };
        if (dlg.ShowDialog() != true) return;
        _inputFolder = dlg.FolderName;
        _imgPaths = ImageFormats.EnumerateImages(_inputFolder).ToList();
        UpdateFileDisplay();
    }

    [RelayCommand]
    private void BrowseOutput()
    {
        var dlg = new OpenFolderDialog { Title = "选择结果保存目录" };
        if (dlg.ShowDialog() != true) return;
        OutputDir = dlg.FolderName;
    }

    [RelayCommand]
    private async Task RunAsync()
    {
        var cfg = _session.Config;
        var yolo = PackLayout.ResolveYolo(_session.AppRoot, cfg.YoloPath);
        if (string.IsNullOrWhiteSpace(yolo) || !File.Exists(yolo))
        {
            MessageBox.Show(
                "未找到 YOLO 权重。请确认程序目录下存在：\ndetect_weights\\best.pt\n\n无需进入设置页，把权重放到上述路径后重新打开即可。",
                "缺少检测模型", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        if (_imgPaths.Count == 0)
        {
            MessageBox.Show("请先选择输入图像。", "提示", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        if (!DetectOnly && !_engine.IsLoaded)
        {
            MessageBox.Show(
                "分类引擎未加载。请确认程序目录下存在：\ncheckpoints\\model.onnx\n以及 python_runtime（完整包）。\n\n可查看 logs\\startup.txt。\n\n若仅需定位坐标，请勾选「仅检测定位」。",
                "引擎未就绪", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        if (DetectOnly && DownsampleEnabled && DownsampleMaxSide < 64)
        {
            MessageBox.Show("下采样目标边长至少为 64。", "提示", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        if (string.IsNullOrWhiteSpace(OutputDir))
        {
            MessageBox.Show("请指定结果保存目录。", "提示", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        Directory.CreateDirectory(OutputDir);
        Rows.Clear();
        UpdateTotal();
        IsRunning = true;
        IsStopping = false;
        _main.IsBusy = true;
        ProgressVisible = true;
        ProgressMaximum = Math.Max(1, _imgPaths.Count);
        ProgressValue = 0;
        StageVisible = true;
        StageText = "准备中…";
        _cts = new CancellationTokenSource();
        var token = _cts.Token;

        var options = new SahiRunOptions
        {
            YoloPath = yolo,
            OutputDir = OutputDir,
            Device = cfg.SahiDevice,
            SliceSize = cfg.SahiSliceSize,
            Overlap = cfg.SahiOverlap,
            DetConf = cfg.SahiDetConf,
            BatchSize = cfg.SahiBatchSize,
            CropPadding = cfg.SahiCropPadding,
            IosThresh = cfg.SahiIosThresh,
            MinAreaRatio = cfg.SahiMinAreaRatio,
            MaxAspectRatio = cfg.SahiMaxAspectRatio,
            EdgeFilter = cfg.SahiEdgeFilter,
            EdgeMarginPx = cfg.SahiEdgeMarginPx,
            DetectOnly = DetectOnly,
            DownsampleEnabled = DetectOnly && DownsampleEnabled,
            DownsampleMaxSide = Math.Max(64, DownsampleMaxSide),
            DownsampleInterpolation = string.IsNullOrWhiteSpace(SelectedInterpolation)
                ? "area"
                : SelectedInterpolation.Trim(),
            SaveVisualization = SaveVisualization,
        };

        var progress = new Progress<SahiProgress>(p =>
        {
            ProgressMaximum = Math.Max(1, p.Total);
            ProgressValue = Math.Min(p.Current, ProgressMaximum);
            StageText = p.Message;
            if (p.LastImage != null)
                AppendRow(p.LastImage);
        });

        var cancelled = false;
        var errorMessage = "";
        IReadOnlyList<SahiImageStats> stats = Array.Empty<SahiImageStats>();
        try
        {
            using var _ = LocalDiagnostics.Measure("sahi.run", $"n={_imgPaths.Count}");
            stats = await _sahi.ProcessImagesAsync(_imgPaths, options, progress, token);
            cancelled = token.IsCancellationRequested;
            StageText = cancelled ? "已停止" : "处理完成";
            _main.StatusText = DetectOnly
                ? $"钻石定位完成：{stats.Count} 张图"
                : $"钻石检测完成：{stats.Count} 张图";
        }
        catch (OperationCanceledException)
        {
            cancelled = true;
            StageText = "已停止";
            _main.StatusText = "钻石检测已停止";
            LocalDiagnostics.Event("sahi.run.cancelled");
        }
        catch (Exception ex)
        {
            LocalDiagnostics.Error("sahi.run", ex);
            errorMessage = UserMessage.Format(ex);
            StageText = "出错";
        }
        finally
        {
            IsRunning = false;
            IsStopping = false;
            ProgressVisible = false;
            StageVisible = false;
            _main.IsBusy = false;
            _cts?.Dispose();
            _cts = null;
            UpdateTotal();
        }

        if (!string.IsNullOrEmpty(errorMessage))
            MessageBox.Show(errorMessage, "处理出错", MessageBoxButton.OK, MessageBoxImage.Error);

        // 完整模式：单张/多图/文件夹批量均写出 summary.csv（与历史表头一致）
        if (!DetectOnly && stats.Count > 0 && !string.IsNullOrWhiteSpace(OutputDir))
            SahiSummaryCsv.Write(OutputDir, stats);

        if (!cancelled
            && !DetectOnly
            && RunUniformityAfter
            && stats.Count > 0
            && !string.IsNullOrWhiteSpace(OutputDir)
            && string.IsNullOrEmpty(errorMessage))
        {
            try
            {
                StageVisible = true;
                StageText = "正在计算均匀度…";
                _main.IsBusy = true;
                _main.StatusText = "检测完成，正在计算均匀度…";
                await _uniformity.RunForOutputRootAsync(OutputDir);
                _main.StatusText = "均匀度已写入各子目录 uniformity_scores.json 与根目录 uniformity_summary.csv";
            }
            catch (Exception ex)
            {
                LocalDiagnostics.Error("uniformity.after_sahi", ex);
                MessageBox.Show(
                    "检测已完成，但均匀度计算失败：\n" + UserMessage.Format(ex),
                    "均匀度", MessageBoxButton.OK, MessageBoxImage.Warning);
            }
            finally
            {
                StageVisible = false;
                _main.IsBusy = false;
            }
        }

        if (stats.Count > 0 && !cancelled)
        {
            var reply = MessageBox.Show(
                $"共处理 {stats.Count} 张图像。\n是否打开结果保存文件夹？",
                "处理完成", MessageBoxButton.YesNo, MessageBoxImage.Question);
            if (reply == MessageBoxResult.Yes && Directory.Exists(OutputDir))
                Process.Start(new ProcessStartInfo { FileName = OutputDir, UseShellExecute = true });
        }
    }

    [RelayCommand]
    private void Stop()
    {
        if (!IsRunning || IsStopping || _cts is null) return;
        IsStopping = true;
        StageText = "正在停止…";
        _cts.Cancel();
    }

    [RelayCommand]
    private void OpenOutput()
    {
        if (Directory.Exists(OutputDir))
            Process.Start(new ProcessStartInfo { FileName = OutputDir, UseShellExecute = true });
    }

    private void AppendRow(SahiImageStats stats)
    {
        Rows.Add(new SahiResultRow
        {
            Image = stats.Image,
            DiamondsText = stats.TotalDiamonds.ToString(),
            Distribution = stats.DefectDistribution,
            TimeText = $"{stats.TotalTimeS:0.00}",
            OutputDir = stats.OutputDir,
        });
        UpdateTotal();
    }

    private void UpdateTotal()
    {
        var diamonds = Rows.Sum(r => int.TryParse(r.DiamondsText, out var n) ? n : 0);
        TotalText = $"合计：{Rows.Count} 张图 · {diamonds} 颗钻石";
    }

    private void UpdateFileDisplay()
    {
        var n = _imgPaths.Count;
        if (n == 0)
        {
            InputPathText = "未选择图像";
            ImageCountText = "";
            return;
        }

        if (!string.IsNullOrEmpty(_inputFolder))
            InputPathText = Path.GetFullPath(_inputFolder);
        else
            InputPathText = string.Join(Environment.NewLine,
                _imgPaths.Select(p => Path.GetFullPath(p)));
        ImageCountText = $"共 {n} 张图像";
    }

    private string ResolveDefaultOutput()
    {
        var raw = _session.Config.SahiOutputDir;
        if (string.IsNullOrWhiteSpace(raw)) raw = "sahi_output";
        return ResolvePath(raw);
    }

    private string ResolvePath(string? relative)
    {
        if (string.IsNullOrWhiteSpace(relative)) return "";
        if (Path.IsPathRooted(relative)) return relative;
        return Path.GetFullPath(Path.Combine(_session.AppRoot, relative));
    }
}
