using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Windows;
using System.Windows.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using DiamondDetect.Core;
using DiamondDetect.Core.Abstractions;
using DiamondDetect.Core.Services;
using DiamondDetect.Wpf.Services;
using Microsoft.Win32;

namespace DiamondDetect.Wpf.ViewModels;

public partial class UniformityTileRow : ObservableObject
{
    [ObservableProperty] private bool isChecked = true;
    public string Image { get; init; } = "";
    public string JsonPath { get; init; } = "";
    public string SourceKind { get; init; } = "";
    public string SourceKindText => SourceKind.Equals("result.json", StringComparison.OrdinalIgnoreCase)
        ? "result"
        : SourceKind.Equals("detect_boxes.json", StringComparison.OrdinalIgnoreCase)
            ? "boxes"
            : SourceKind;
}

public partial class UniformityResultRow : ObservableObject
{
    public string Image { get; init; } = "";
    public string SourceJson { get; init; } = "";
    public string VisPath { get; set; } = "";
    public string NPointsText { get; init; } = "";
    public string VoronoiText { get; init; } = "";
    public string NnText { get; init; } = "";
    public string ClarkEvansText { get; init; } = "";
    public string DelaunayText { get; init; } = "";
    public string GridText { get; init; } = "";
    public string Status { get; init; } = "";
    public string ConfFilter { get; init; } = "";
}

public partial class UniformityViewModel : ObservableObject
{
    private readonly AppSession _session;
    private readonly IUniformityAnalyzer _analyzer;
    private readonly IConfigService _configService;
    private readonly MainViewModel _main;
    private CancellationTokenSource? _cts;

    public UniformityViewModel(
        AppSession session,
        IUniformityAnalyzer analyzer,
        IConfigService configService,
        MainViewModel main)
    {
        _session = session;
        _analyzer = analyzer;
        _configService = configService;
        _main = main;
        OutputRoot = ResolveDefaultOutput();
        ConfThresholdText = _session.Config.UniformityConfThreshold.ToString("0.##", CultureInfo.InvariantCulture);
    }

    public ObservableCollection<UniformityTileRow> Tiles { get; } = new();
    public ObservableCollection<UniformityResultRow> Rows { get; } = new();

    [ObservableProperty] private string pageHintText =
        "产品输出根：优先 detect_boxes.json，否则 result.json。点选结果行可预览/生成网格热力可视化（uniformity_vis.jpg）。";
    [ObservableProperty] private string outputRoot = "";
    [ObservableProperty] private string confThresholdText = "0.25";
    [ObservableProperty] private string tileListText = "尚未扫描";
    [ObservableProperty] private bool writeVisualization;
    [ObservableProperty] private bool isScanning;
    [ObservableProperty] private bool isRunning;
    [ObservableProperty] private bool isStopping;
    [ObservableProperty] private bool isRenderingVis;
    [ObservableProperty] private bool progressVisible;
    [ObservableProperty] private int progressValue;
    [ObservableProperty] private int progressMaximum = 1;
    [ObservableProperty] private string stageText = "";
    [ObservableProperty] private bool stageVisible;
    [ObservableProperty] private string summaryText = "";
    [ObservableProperty] private string? lastSummaryCsv;
    [ObservableProperty] private UniformityResultRow? selectedRow;
    [ObservableProperty] private BitmapImage? previewImage;
    [ObservableProperty] private string previewMeta = "点选结果行查看可视化";

    public bool CanStop => IsRunning && !IsStopping;
    public bool CanRun => !IsRunning && !IsScanning && !IsRenderingVis;
    public bool CanRegenerateVis => CanRun && SelectedRow != null && !string.IsNullOrWhiteSpace(SelectedRow.SourceJson);

    partial void OnIsRunningChanged(bool value)
    {
        OnPropertyChanged(nameof(CanStop));
        OnPropertyChanged(nameof(CanRun));
        OnPropertyChanged(nameof(CanRegenerateVis));
    }

    partial void OnIsStoppingChanged(bool value) => OnPropertyChanged(nameof(CanStop));
    partial void OnIsScanningChanged(bool value)
    {
        OnPropertyChanged(nameof(CanRun));
        OnPropertyChanged(nameof(CanRegenerateVis));
    }

    partial void OnIsRenderingVisChanged(bool value)
    {
        OnPropertyChanged(nameof(CanRun));
        OnPropertyChanged(nameof(CanRegenerateVis));
    }

    partial void OnSelectedRowChanged(UniformityResultRow? value)
    {
        OnPropertyChanged(nameof(CanRegenerateVis));
        _ = LoadOrOfferPreviewAsync(value);
    }

    [RelayCommand]
    private async Task BrowseOutputRootAsync()
    {
        var dlg = new OpenFolderDialog { Title = "选择产品检测输出根目录（含各子图目录）" };
        if (dlg.ShowDialog() != true) return;
        OutputRoot = dlg.FolderName;
        await ScanTilesAsync();
    }

    [RelayCommand]
    private void OpenOutputRoot()
    {
        if (Directory.Exists(OutputRoot))
            Process.Start(new ProcessStartInfo { FileName = OutputRoot, UseShellExecute = true });
    }

    [RelayCommand]
    private async Task ScanTilesAsync()
    {
        if (string.IsNullOrWhiteSpace(OutputRoot) || !Directory.Exists(OutputRoot))
        {
            MessageBox.Show("请先选择有效的检测输出根目录。", "提示", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        IsScanning = true;
        _main.IsBusy = true;
        StageVisible = true;
        StageText = "正在扫描子图…";
        Tiles.Clear();
        try
        {
            var list = await _analyzer.ListTilesAsync(OutputRoot);
            foreach (var t in list)
            {
                Tiles.Add(new UniformityTileRow
                {
                    IsChecked = true,
                    Image = t.Image,
                    JsonPath = t.JsonPath,
                    SourceKind = t.SourceKind,
                });
            }
            TileListText = Tiles.Count == 0
                ? "未找到 detect_boxes.json / result.json"
                : $"共 {Tiles.Count} 张 · 已全选";
            _main.StatusText = $"均匀度：已扫描 {Tiles.Count} 张子图";
            StageText = TileListText;
        }
        catch (Exception ex)
        {
            LocalDiagnostics.Error("uniformity.scan", ex);
            MessageBox.Show(UserMessage.Format(ex), "扫描失败", MessageBoxButton.OK, MessageBoxImage.Error);
            TileListText = "扫描失败";
        }
        finally
        {
            IsScanning = false;
            _main.IsBusy = false;
            StageVisible = false;
        }
    }

    [RelayCommand]
    private void SelectAllTiles()
    {
        foreach (var t in Tiles)
            t.IsChecked = true;
        UpdateTileListText();
    }

    [RelayCommand]
    private void SelectNoneTiles()
    {
        foreach (var t in Tiles)
            t.IsChecked = false;
        UpdateTileListText();
    }

    [RelayCommand]
    private async Task RunAsync()
    {
        if (string.IsNullOrWhiteSpace(OutputRoot) || !Directory.Exists(OutputRoot))
        {
            MessageBox.Show("请先选择有效的检测输出根目录。", "提示", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        if (!TryParseConf(out var conf))
        {
            MessageBox.Show("置信度阈值须为 0～1 之间的数字。", "提示", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        if (Tiles.Count == 0)
            await ScanTilesAsync();

        var selected = Tiles.Where(t => t.IsChecked).Select(t => t.JsonPath).Where(p => !string.IsNullOrWhiteSpace(p)).ToList();
        if (selected.Count == 0)
        {
            MessageBox.Show("请至少勾选一张子图。", "提示", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        PersistConf(conf);

        Rows.Clear();
        SelectedRow = null;
        PreviewImage = null;
        PreviewMeta = "点选结果行查看可视化";
        SummaryText = "";
        LastSummaryCsv = null;
        IsRunning = true;
        IsStopping = false;
        _main.IsBusy = true;
        ProgressVisible = true;
        ProgressMaximum = Math.Max(1, selected.Count);
        ProgressValue = 0;
        StageVisible = true;
        StageText = "准备中…";
        _cts = new CancellationTokenSource();
        var token = _cts.Token;

        var progress = new Progress<(int current, int total, string message)>(p =>
        {
            ProgressMaximum = Math.Max(1, p.total);
            ProgressValue = Math.Min(p.current, ProgressMaximum);
            StageText = p.message;
        });

        var cancelled = false;
        var errorMessage = "";
        UniformityBatchResult? batch = null;
        try
        {
            using var _ = LocalDiagnostics.Measure("uniformity.run", $"n={selected.Count};vis={WriteVisualization}");
            batch = await _analyzer.AnalyzeOutputRootAsync(
                OutputRoot, conf, selected, WriteVisualization, progress, token);
            cancelled = token.IsCancellationRequested;
            foreach (var row in batch.Rows)
                Rows.Add(MapRow(row));
            LastSummaryCsv = batch.SummaryCsv;
            var ok = batch.Rows.Count(r =>
                string.Equals(r.Status, "ok", StringComparison.OrdinalIgnoreCase)
                || string.Equals(r.Status, "样本不足", StringComparison.Ordinal));
            SummaryText = cancelled
                ? $"已停止：已处理 {batch.Rows.Count} 项"
                : $"完成：{batch.Rows.Count} 项 · 有效/样本不足 {ok} · summary 已写"
                  + (WriteVisualization ? " · 已批量出可视化" : "");
            StageText = cancelled ? "已停止" : "处理完成";
            _main.StatusText = cancelled ? "均匀度分析已停止" : $"均匀度分析完成：{batch.Rows.Count} 项";
            if (Rows.Count > 0)
                SelectedRow = Rows[0];
        }
        catch (OperationCanceledException)
        {
            cancelled = true;
            StageText = "已停止";
            _main.StatusText = "均匀度分析已停止";
        }
        catch (Exception ex)
        {
            LocalDiagnostics.Error("uniformity.run", ex);
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
        }

        if (!string.IsNullOrEmpty(errorMessage))
            MessageBox.Show(errorMessage, "均匀度分析出错", MessageBoxButton.OK, MessageBoxImage.Error);
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
    private async Task RegenerateVisAsync()
    {
        if (SelectedRow is null || string.IsNullOrWhiteSpace(SelectedRow.SourceJson))
            return;
        if (!TryParseConf(out var conf))
        {
            MessageBox.Show("置信度阈值须为 0～1 之间的数字。", "提示", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        PersistConf(conf);
        IsRenderingVis = true;
        _main.IsBusy = true;
        PreviewMeta = "正在生成可视化…";
        try
        {
            var vis = await _analyzer.RenderVisualizationAsync(SelectedRow.SourceJson, conf);
            if (string.IsNullOrWhiteSpace(vis.VisPath) || !File.Exists(vis.VisPath))
            {
                PreviewImage = null;
                PreviewMeta = string.IsNullOrWhiteSpace(vis.Status) ? "可视化未生成" : vis.Status;
                return;
            }

            SelectedRow.VisPath = vis.VisPath;
            PreviewImage = ThumbnailCache.Shared.GetOrLoad(vis.VisPath, 720);
            var bg = string.IsNullOrEmpty(vis.Background) ? "白底" : vis.Background;
            PreviewMeta = $"{SelectedRow.Image} · {bg} · n={vis.NPoints}";
            _main.StatusText = $"已生成 {Path.GetFileName(vis.VisPath)}";
        }
        catch (Exception ex)
        {
            LocalDiagnostics.Error("uniformity.vis", ex);
            PreviewImage = null;
            PreviewMeta = "可视化失败";
            MessageBox.Show(UserMessage.Format(ex), "可视化", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
        finally
        {
            IsRenderingVis = false;
            _main.IsBusy = false;
        }
    }

    [RelayCommand]
    private void OpenVisFile()
    {
        var path = SelectedRow?.VisPath;
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            MessageBox.Show("当前行尚无可视化文件，请先点「重新生成」。", "提示", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }
        Process.Start(new ProcessStartInfo { FileName = path, UseShellExecute = true });
    }

    /// <summary>供检测页联动：扫描并全选后计算（不切换导航；默认不出可视化）。</summary>
    public async Task RunForOutputRootAsync(string outputRoot, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(outputRoot) || !Directory.Exists(outputRoot))
            return;

        var conf = _session.Config.UniformityConfThreshold;
        ConfThresholdText = conf.ToString("0.##", CultureInfo.InvariantCulture);
        OutputRoot = outputRoot;

        var list = await _analyzer.ListTilesAsync(outputRoot, cancellationToken);
        var paths = list.Select(t => t.JsonPath).ToList();

        Application.Current.Dispatcher.Invoke(() =>
        {
            Tiles.Clear();
            foreach (var t in list)
            {
                Tiles.Add(new UniformityTileRow
                {
                    IsChecked = true,
                    Image = t.Image,
                    JsonPath = t.JsonPath,
                    SourceKind = t.SourceKind,
                });
            }
            UpdateTileListText();
        });

        var batch = await _analyzer.AnalyzeOutputRootAsync(
            outputRoot, conf, paths, writeVisualization: false, progress: null, cancellationToken);
        Application.Current.Dispatcher.Invoke(() =>
        {
            Rows.Clear();
            foreach (var row in batch.Rows)
                Rows.Add(MapRow(row));
            LastSummaryCsv = batch.SummaryCsv;
            SummaryText = $"联动完成：{batch.Rows.Count} 项";
            if (Rows.Count > 0)
                SelectedRow = Rows[0];
        });
    }

    private async Task LoadOrOfferPreviewAsync(UniformityResultRow? row)
    {
        if (row is null)
        {
            PreviewImage = null;
            PreviewMeta = "点选结果行查看可视化";
            return;
        }

        var existing = row.VisPath;
        if (string.IsNullOrWhiteSpace(existing) && !string.IsNullOrWhiteSpace(row.SourceJson))
        {
            var candidate = Path.Combine(Path.GetDirectoryName(row.SourceJson) ?? "", "uniformity_vis.jpg");
            if (File.Exists(candidate))
            {
                existing = candidate;
                row.VisPath = candidate;
            }
        }

        if (!string.IsNullOrWhiteSpace(existing) && File.Exists(existing))
        {
            PreviewImage = ThumbnailCache.Shared.GetOrLoad(existing, 720);
            PreviewMeta = $"{row.Image} · {Path.GetFileName(existing)}";
            return;
        }

        PreviewImage = null;
        PreviewMeta = $"{row.Image} · 尚无可视化，可点「重新生成」";
        // 按需自动生成（点选即出图）
        if (!string.IsNullOrWhiteSpace(row.SourceJson) && CanRun)
            await RegenerateVisAsync();
    }

    private void UpdateTileListText()
    {
        var n = Tiles.Count;
        var c = Tiles.Count(t => t.IsChecked);
        TileListText = n == 0 ? "尚未扫描" : $"共 {n} 张 · 已选 {c}";
    }

    private bool TryParseConf(out double conf)
    {
        conf = 0.25;
        if (!double.TryParse(ConfThresholdText.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out conf)
            && !double.TryParse(ConfThresholdText.Trim(), NumberStyles.Float, CultureInfo.CurrentCulture, out conf))
            return false;
        return conf is >= 0 and <= 1;
    }

    private void PersistConf(double conf)
    {
        var c = _configService.Current;
        if (Math.Abs(c.UniformityConfThreshold - conf) < 1e-9)
            return;
        c.UniformityConfThreshold = conf;
        _configService.Save(c);
        _session.Config = c;
    }

    private static UniformityResultRow MapRow(UniformityScoreRow row) => new()
    {
        Image = row.Image,
        SourceJson = row.SourceJson,
        VisPath = row.VisPath,
        NPointsText = row.NPoints.ToString(CultureInfo.InvariantCulture),
        VoronoiText = Fmt(row.VoronoiAreaCvNormalized),
        NnText = Fmt(row.NnDistanceCvNormalized),
        ClarkEvansText = Fmt(row.ClarkEvansR),
        DelaunayText = Fmt(row.DelaunayEdgeCvNormalized),
        GridText = Fmt(row.GridDensityCv),
        Status = row.Status,
        ConfFilter = row.ConfFilter,
    };

    private static string Fmt(double? v) =>
        v is null ? "" : v.Value.ToString("0.####", CultureInfo.InvariantCulture);

    private string ResolveDefaultOutput()
    {
        var raw = _session.Config.SahiOutputDir;
        if (string.IsNullOrWhiteSpace(raw)) raw = "sahi_output";
        if (Path.IsPathRooted(raw)) return raw;
        return Path.GetFullPath(Path.Combine(_session.AppRoot, raw));
    }
}
