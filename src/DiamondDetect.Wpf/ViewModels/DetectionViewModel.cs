using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using System.Windows.Media;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using DiamondDetect.Core;
using DiamondDetect.Core.Abstractions;
using DiamondDetect.Core.Models;
using DiamondDetect.Core.Services;
using DiamondDetect.Wpf.Services;
using Microsoft.Win32;

namespace DiamondDetect.Wpf.ViewModels;

public partial class DetectionViewModel : ObservableObject
{
    private readonly AppSession _session;
    private readonly IInferenceEngine _engine;
    private readonly ResultStore _results;
    private readonly ExportService _export;
    private readonly MainViewModel _main;
    private CancellationTokenSource? _batchCts;
    private DetectionResult? _lastSingle;

    public DetectionViewModel(
        AppSession session,
        IInferenceEngine engine,
        ResultStore results,
        ExportService export,
        MainViewModel main)
    {
        _session = session;
        _engine = engine;
        _results = results;
        _export = export;
        _main = main;
    }

    public ObservableCollection<ScoreBarItem> ScoreBars { get; } = new();
    public ObservableCollection<BatchRowItem> BatchRows { get; } = new();

    [ObservableProperty] private string? singleImagePath;
    [ObservableProperty] private ImageSource? singlePreview;
    [ObservableProperty] private string predictedClass = "—";
    [ObservableProperty] private string confidenceText = "置信度: —";
    [ObservableProperty] private string elapsedText = "";
    [ObservableProperty] private Brush predictedClassBrush = new SolidColorBrush(Color.FromRgb(0x15, 0x65, 0xC0));
    [ObservableProperty] private bool canExportSingle;
    [ObservableProperty] private bool isSingleBusy;

    [ObservableProperty] private string folderPath = "";
    [ObservableProperty] private string folderInfo = "未选择文件夹";
    [ObservableProperty] private bool isBatchRunning;
    [ObservableProperty] private int batchProgress;
    [ObservableProperty] private int batchMaximum = 1;
    [ObservableProperty] private bool batchProgressVisible;
    [ObservableProperty] private bool canExportBatch;

    [RelayCommand]
    private void BrowseSingleImage()
    {
        var dlg = new OpenFileDialog
        {
            Filter = "图像|*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff;*.webp|所有文件|*.*",
        };
        if (dlg.ShowDialog() == true)
            SetSingleImage(dlg.FileName);
    }

    public void SetSingleImage(string path)
    {
        if (!File.Exists(path) || !ImageFormats.IsImage(path))
            return;
        SingleImagePath = path;
        SinglePreview = ThumbnailCache.Shared.GetOrLoad(path, 560);
    }

    [RelayCommand]
    private async Task RunSingleAsync()
    {
        if (string.IsNullOrWhiteSpace(SingleImagePath))
        {
            UserMessage.Warn("提示", "请先选择或拖入一张图像。");
            return;
        }
        if (!_engine.IsLoaded)
        {
            UserMessage.Warn("提示", "模型未加载，请前往「设置」页面加载模型。");
            return;
        }

        try
        {
            IsSingleBusy = true;
            _main.IsBusy = true;
            using var _ = LocalDiagnostics.Measure("detect.single");
            var path = SingleImagePath;
            var r = await Task.Run(() => _engine.Predict(path));
            _lastSingle = r;
            _results.Upsert(r);
            ShowSingle(r);
            CanExportSingle = true;
            _main.StatusText = $"单张检测完成：{r.Class} ({r.ConfidenceText})";
        }
        catch (Exception ex)
        {
            UserMessage.Error("检测出错", ex);
        }
        finally
        {
            IsSingleBusy = false;
            _main.IsBusy = false;
        }
    }

    private void ShowSingle(DetectionResult r)
    {
        PredictedClass = string.IsNullOrEmpty(r.Class) ? "—" : r.Class;
        var conf = r.Confidence;
        PredictedClassBrush = new SolidColorBrush(
            conf >= 0.8 ? Color.FromRgb(0x1B, 0x5E, 0x20)
            : conf >= 0.5 ? Color.FromRgb(0xE6, 0x51, 0x00)
            : Color.FromRgb(0xB7, 0x1C, 0x1C));

        var text = $"置信度：{conf * 100:0.0}%";
        ConfidenceText = text;
        ElapsedText = $"推理耗时：{r.ElapsedMs:0.0} ms";

        ScoreBars.Clear();
        foreach (var kv in r.AllScores.OrderByDescending(x => x.Value))
        {
            ScoreBars.Add(new ScoreBarItem
            {
                Name = kv.Key,
                Score = kv.Value,
                IsPredicted = kv.Key == r.Class,
            });
        }
    }

    [RelayCommand]
    private void BrowseFolder()
    {
        var dlg = new OpenFolderDialog { Title = "选择图像文件夹" };
        if (dlg.ShowDialog() != true) return;
        FolderPath = dlg.FolderName;
        var n = ImageFormats.EnumerateImages(FolderPath).Count;
        FolderInfo = $"共 {n} 张图像";
    }

    [RelayCommand]
    private async Task RunBatchAsync()
    {
        if (string.IsNullOrWhiteSpace(FolderPath) || !Directory.Exists(FolderPath))
        {
            UserMessage.Warn("提示", "请先选择有效的文件夹。");
            return;
        }
        if (!_engine.IsLoaded)
        {
            UserMessage.Warn("提示", "模型未加载，请前往「设置」页面加载模型。");
            return;
        }

        var imgs = ImageFormats.EnumerateImages(FolderPath);
        if (imgs.Count == 0)
        {
            UserMessage.Warn("提示", "所选文件夹中没有支持的图像文件。");
            return;
        }

        BatchRows.Clear();
        _results.Clear();
        CanExportBatch = false;
        BatchMaximum = imgs.Count;
        BatchProgress = 0;
        BatchProgressVisible = true;
        IsBatchRunning = true;
        _main.IsBusy = true;
        _batchCts = new CancellationTokenSource();
        var token = _batchCts.Token;
        var progress = new Progress<(int current, int total)>(p =>
        {
            BatchProgress = p.current;
            BatchMaximum = Math.Max(1, p.total);
        });

        try
        {
            using var _ = LocalDiagnostics.Measure("detect.batch", $"n={imgs.Count}");
            var list = await Task.Run(() => _engine.PredictBatch(imgs, progress, token), token);
            foreach (var r in list)
            {
                _results.Upsert(r);
                BatchRows.Add(new BatchRowItem
                {
                    FileName = r.FileName,
                    Class = r.Class,
                    ConfidenceText = r.ConfidenceText,
                    ElapsedText = $"{r.ElapsedMs:0.0}",
                });
            }
            FolderInfo = $"完成 {list.Count} 张";
            CanExportBatch = list.Count > 0;
            _main.StatusText = $"批量检测完成：{list.Count} 张";
        }
        catch (OperationCanceledException)
        {
            FolderInfo = $"已停止（当前 {BatchRows.Count} 张）";
            CanExportBatch = BatchRows.Count > 0;
            _main.StatusText = "批量检测已停止";
            LocalDiagnostics.Event("detect.batch.cancelled", $"done={BatchRows.Count}");
        }
        catch (Exception ex)
        {
            UserMessage.Error("批量检测出错", ex);
        }
        finally
        {
            IsBatchRunning = false;
            BatchProgressVisible = false;
            _main.IsBusy = false;
            _batchCts?.Dispose();
            _batchCts = null;
        }
    }

    [RelayCommand]
    private void StopBatch()
    {
        _batchCts?.Cancel();
    }

    [RelayCommand]
    private void GoResults() => _main.NavigateTo(MainViewModel.NavResults);

    [RelayCommand]
    private void ExportSingle()
    {
        if (_lastSingle is null)
        {
            UserMessage.Warn("提示", "请先完成单张检测。");
            return;
        }
        ExportClassified(new[] { _lastSingle }, onlyChecked: false,
            Path.GetFileName(Path.GetDirectoryName(_lastSingle.Path) ?? "导出结果"));
    }

    [RelayCommand]
    private void ExportBatch()
    {
        if (_results.Count == 0)
        {
            UserMessage.Warn("提示", "请先完成批量检测。");
            return;
        }
        var name = string.IsNullOrWhiteSpace(FolderPath) ? "导出结果" : Path.GetFileName(FolderPath);
        ExportClassified(_results.Items, onlyChecked: false, name);
    }

    private void ExportClassified(IEnumerable<DetectionResult> results, bool onlyChecked, string defaultName)
    {
        var parentDlg = new OpenFolderDialog { Title = "选择导出位置（上级目录）" };
        if (parentDlg.ShowDialog() != true) return;

        var safe = ExportService.SanitizeFolderName(defaultName);
        var ask = new Views.InputPromptWindow(
            "导出文件夹名称",
            $"将在以下目录下创建子文件夹：\n{parentDlg.FolderName}\n\n文件夹名称（可修改）：",
            safe)
        { Owner = Application.Current.MainWindow };
        if (ask.ShowDialog() != true) return;

        var exportRoot = Path.Combine(parentDlg.FolderName, ask.Value);
        try
        {
            var n = _export.ExportClassifiedImages(results, exportRoot, onlyChecked);
            UserMessage.Info("导出完成", $"已导出 {n} 张到：\n{exportRoot}");
        }
        catch (Exception ex)
        {
            UserMessage.Error("导出失败", ex);
        }
    }
}
