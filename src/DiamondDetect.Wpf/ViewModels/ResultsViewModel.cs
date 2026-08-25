using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using DiamondDetect.Core;
using DiamondDetect.Core.Abstractions;
using DiamondDetect.Core.Models;
using DiamondDetect.Core.Services;
using DiamondDetect.Wpf.Services;
using Microsoft.Win32;

namespace DiamondDetect.Wpf.ViewModels;

public partial class ResultRowItem : ObservableObject
{
    private DetectionResult _result = null!;

    public int StoreIndex { get; set; }

    public DetectionResult Result
    {
        get => _result;
        set
        {
            if (ReferenceEquals(_result, value)) return;
            _result = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(FileName));
            OnPropertyChanged(nameof(Class));
            OnPropertyChanged(nameof(ConfidenceText));
            OnPropertyChanged(nameof(Confidence));
        }
    }

    public string FileName => Result.FileName;
    public string Class => Result.Class;
    public string ConfidenceText => Result.ConfidenceText;
    public double Confidence => Result.Confidence;

    [ObservableProperty]
    private bool isChecked = true;

    [ObservableProperty]
    private BitmapImage? thumbnail;

    public void Bind(int storeIndex, DetectionResult result, bool isChecked)
    {
        var pathChanged = _result is null
            || !string.Equals(_result.Path, result.Path, StringComparison.OrdinalIgnoreCase);
        StoreIndex = storeIndex;
        Result = result;
        IsChecked = isChecked;
        if (pathChanged)
            Thumbnail = null;
    }
}

public partial class ResultsViewModel : ObservableObject
{
    private readonly AppSession _session;
    private readonly ResultStore _store;
    private readonly ExportService _export;
    private readonly IInferenceEngine _engine;
    private readonly MainViewModel _main;
    private string? _confSort; // null | asc | desc
    private int _thumbGeneration;

    public ResultsViewModel(
        AppSession session,
        ResultStore store,
        ExportService export,
        IInferenceEngine engine,
        MainViewModel main)
    {
        _session = session;
        _store = store;
        _export = export;
        _engine = engine;
        _main = main;
        _store.Changed += (_, _) => Application.Current.Dispatcher.Invoke(Refresh);
        ClassFilters.Add("全部");
    }

    public ObservableCollection<ResultRowItem> Rows { get; } = new();
    public ObservableCollection<string> ClassFilters { get; } = new();
    public ObservableCollection<string> ArchiveClasses { get; } = new();

    [ObservableProperty] private string selectedClassFilter = "全部";
    [ObservableProperty] private double maxConfidenceFilter = 1.0;
    [ObservableProperty] private string statsText = "共 0 条";
    [ObservableProperty] private string bottomText = "";
    [ObservableProperty] private string confidenceHeader = "置信度 ↕";
    [ObservableProperty] private string toggleSelectText = "全不选";
    [ObservableProperty] private string archiveHint = "勾选或选中图像 → 点击正确类别归档";
    [ObservableProperty] private BitmapImage? previewImage;
    [ObservableProperty] private string previewMeta = "";
    [ObservableProperty] private ResultRowItem? selectedRow;

    private bool _allChecked = true;
    private bool _suppressFilterRefresh;

    partial void OnSelectedClassFilterChanged(string value)
    {
        if (!_suppressFilterRefresh) Refresh();
    }

    partial void OnMaxConfidenceFilterChanged(double value)
    {
        if (!_suppressFilterRefresh) Refresh();
    }

    partial void OnSelectedRowChanged(ResultRowItem? value)
    {
        if (value is null) return;
        ArchiveHint =
            $"当前：{value.FileName}\n预测 {value.Class} ({value.ConfidenceText}) → 点击正确类别归档";
        PreviewMeta = MetaFor(value.Result);
        LoadPreview(value.Result.Path);
    }

    public void Refresh()
    {
        var items = _store.Items;
        var visible = items.Where(r => r.IsResultsListItem).ToList();

        var classes = visible
            .Select(r => r.Class)
            .Where(c => !string.IsNullOrEmpty(c) && c != "ERROR")
            .Distinct()
            .OrderBy(c => c)
            .ToList();

        var prevFilter = SelectedClassFilter;
        _suppressFilterRefresh = true;
        try
        {
            SyncFilterList(ClassFilters, classes, prevFilter);
        }
        finally
        {
            _suppressFilterRefresh = false;
        }

        SyncStringList(ArchiveClasses, _engine.Classes);

        var filtered = visible
            .Where(r =>
                (SelectedClassFilter == "全部" || r.Class == SelectedClassFilter)
                && r.Confidence <= MaxConfidenceFilter)
            .ToList();

        if (_confSort == "desc")
            filtered = filtered.OrderByDescending(r => r.Confidence).ToList();
        else if (_confSort == "asc")
            filtered = filtered.OrderBy(r => r.Confidence).ToList();

        foreach (var row in Rows)
        {
            var live = _store.GetAt(row.StoreIndex);
            if (live != null)
                live.IsChecked = row.IsChecked;
        }

        var desired = filtered
            .Select(r => (Idx: _store.IndexOf(r), Result: r))
            .Where(x => x.Idx >= 0)
            .ToList();

        var selectedPath = SelectedRow?.Result.Path;
        ApplyRowsIncremental(desired);

        if (!string.IsNullOrEmpty(selectedPath))
        {
            SelectedRow = Rows.FirstOrDefault(r =>
                string.Equals(r.Result.Path, selectedPath, StringComparison.OrdinalIgnoreCase));
        }

        ScheduleThumbnailLoad(Rows.Where(r => r.Thumbnail is null).ToList());

        StatsText = $"本页 {visible.Count} 条  |  修正页待处理 {_store.FlaggedPendingCount} 项";
        var sortHint = _confSort switch
        {
            "desc" => "  |  排序：置信度 高→低",
            "asc" => "  |  排序：置信度 低→高",
            _ => "",
        };
        BottomText = $"显示 {filtered.Count} 条{sortHint}";
        ConfidenceHeader = _confSort switch
        {
            "desc" => "置信度 ▼",
            "asc" => "置信度 ▲",
            _ => "置信度 ↕",
        };
    }

    /// <summary>
    /// 按 StoreIndex 复用行对象（保留缩略图），仅增删/移动/更新，避免筛选时整表重建。
    /// </summary>
    private void ApplyRowsIncremental(List<(int Idx, DetectionResult Result)> desired)
    {
        var desiredIdx = desired.Select(d => d.Idx).ToHashSet();
        for (var i = Rows.Count - 1; i >= 0; i--)
        {
            if (!desiredIdx.Contains(Rows[i].StoreIndex))
                Rows.RemoveAt(i);
        }

        var byIndex = Rows.ToDictionary(r => r.StoreIndex);
        for (var i = 0; i < desired.Count; i++)
        {
            var (idx, result) = desired[i];
            if (byIndex.TryGetValue(idx, out var row))
            {
                row.Bind(idx, result, result.IsChecked);
                var cur = Rows.IndexOf(row);
                if (cur >= 0 && cur != i)
                    Rows.Move(cur, i);
            }
            else
            {
                row = new ResultRowItem();
                row.Bind(idx, result, result.IsChecked);
                Rows.Insert(i, row);
                byIndex[idx] = row;
            }
        }
    }

    private void SyncFilterList(ObservableCollection<string> target, List<string> classes, string prevFilter)
    {
        var desired = new List<string> { "全部" };
        desired.AddRange(classes);

        for (var i = target.Count - 1; i >= 0; i--)
        {
            if (!desired.Contains(target[i]))
                target.RemoveAt(i);
        }

        for (var i = 0; i < desired.Count; i++)
        {
            var item = desired[i];
            var cur = target.IndexOf(item);
            if (cur < 0)
                target.Insert(i, item);
            else if (cur != i)
                target.Move(cur, i);
        }

        SelectedClassFilter = target.Contains(prevFilter) ? prevFilter : "全部";
    }

    private static void SyncStringList(ObservableCollection<string> target, IReadOnlyList<string> source)
    {
        for (var i = target.Count - 1; i >= 0; i--)
        {
            if (!source.Contains(target[i]))
                target.RemoveAt(i);
        }

        for (var i = 0; i < source.Count; i++)
        {
            var item = source[i];
            var cur = target.IndexOf(item);
            if (cur < 0)
                target.Insert(i, item);
            else if (cur != i)
                target.Move(cur, i);
        }
    }

    private void ScheduleThumbnailLoad(IReadOnlyList<ResultRowItem> rows)
    {
        if (rows.Count == 0) return;
        var gen = ++_thumbGeneration;
        var dispatcher = Application.Current.Dispatcher;
        foreach (var row in rows)
        {
            dispatcher.BeginInvoke(DispatcherPriority.Background, () =>
            {
                if (gen != _thumbGeneration) return;
                row.Thumbnail = ThumbnailCache.Shared.GetOrLoad(row.Result.Path, 96);
            });
        }
    }

    [RelayCommand]
    private void ToggleConfidenceSort()
    {
        _confSort = _confSort == "desc" ? "asc" : "desc";
        Refresh();
    }

    [RelayCommand]
    private void ToggleSelectAll()
    {
        _allChecked = !_allChecked;
        ToggleSelectText = _allChecked ? "全不选" : "全选";
        foreach (var row in Rows)
        {
            row.IsChecked = _allChecked;
            var live = _store.GetAt(row.StoreIndex);
            if (live != null) live.IsChecked = _allChecked;
        }
    }

    [RelayCommand]
    private void SendSelectedToCorrection()
    {
        SyncChecks();
        var indices = Rows.Where(r => r.IsChecked).Select(r => r.StoreIndex).ToList();
        if (indices.Count == 0)
        {
            UserMessage.Info("提示", "请先勾选要送入修正页的图像。");
            return;
        }
        _store.FlagMany(indices);
        _main.StatusText = $"已将 {indices.Count} 项送入误分类修正页（共 {_store.FlaggedPendingCount} 项待处理）";
        LocalDiagnostics.Event("results.send_correction", $"count={indices.Count}");
    }

    [RelayCommand]
    private void SendVisibleToCorrection()
    {
        SyncChecks();
        var indices = Rows.Select(r => r.StoreIndex).ToList();
        if (indices.Count == 0)
        {
            UserMessage.Info("提示", "当前筛选下没有可送入修正页的条目。");
            return;
        }
        _store.FlagMany(indices);
        _main.StatusText = $"已将当前筛选 {indices.Count} 项送入误分类修正页（共 {_store.FlaggedPendingCount} 项待处理）";
        LocalDiagnostics.Event("results.send_visible_correction", $"count={indices.Count}");
    }

    [RelayCommand]
    private void SendOneToCorrection(ResultRowItem? row)
    {
        if (row is null) return;
        SyncChecks();
        _store.FlagAt(row.StoreIndex);
        _main.StatusText = $"「{row.FileName}」已送入误分类修正页（共 {_store.FlaggedPendingCount} 项待处理）";
    }

    [RelayCommand]
    private void ArchiveAsClass(string? trueClass)
    {
        if (string.IsNullOrWhiteSpace(trueClass)) return;
        SyncChecks();
        var indices = Rows.Where(r => r.IsChecked).Select(r => r.StoreIndex).ToList();
        if (indices.Count == 0 && SelectedRow != null)
            indices.Add(SelectedRow.StoreIndex);
        if (indices.Count == 0)
        {
            UserMessage.Info("提示", "请先勾选或选中要归档的图像，再点击正确类别。");
            return;
        }

        if (indices.Count > 1)
        {
            var reply = MessageBox.Show(
                $"将 {indices.Count} 张图像归档为「{trueClass}」并移出结果列表？\n文件将复制到 corrections/{trueClass}/",
                "确认归档", MessageBoxButton.YesNo, MessageBoxImage.Question);
            if (reply != MessageBoxResult.Yes) return;
        }

        var corrRoot = Path.Combine(_session.AppRoot, _session.Config.CorrectionsDir);
        var saved = _export.ArchiveToCorrections(_store, indices, corrRoot, trueClass);
        if (saved == 0)
        {
            UserMessage.Warn("归档失败", "未能保存任何图像，请检查文件是否存在。");
            return;
        }
        _main.StatusText = $"已归档 {saved} 张至 {_session.Config.CorrectionsDir}/{trueClass}/";
        LocalDiagnostics.Event("results.archive", $"class={trueClass};saved={saved}");
        Refresh();
    }

    [RelayCommand]
    private void ExportClassified()
    {
        SyncChecks();
        var parentDlg = new OpenFolderDialog { Title = "选择导出位置（上级目录）" };
        if (parentDlg.ShowDialog() != true) return;
        var ask = new Views.InputPromptWindow(
            "导出文件夹名称",
            $"将在以下目录下创建子文件夹：\n{parentDlg.FolderName}\n\n文件夹名称（可修改）：",
            "分类导出")
        { Owner = Application.Current.MainWindow };
        if (ask.ShowDialog() != true) return;

        var exportRoot = Path.Combine(parentDlg.FolderName, ask.Value);
        try
        {
            var n = _export.ExportClassifiedImages(_store.Items, exportRoot, onlyChecked: true);
            UserMessage.Info("导出完成", $"已导出 {n} 张到：\n{exportRoot}");
            LocalDiagnostics.Event("results.export_classified", $"n={n}");
        }
        catch (Exception ex)
        {
            UserMessage.Error("导出失败", ex);
        }
    }

    [RelayCommand]
    private void ExportCsv()
    {
        var dlg = new SaveFileDialog
        {
            Filter = "CSV|*.csv",
            FileName = "detection_results.csv",
        };
        if (dlg.ShowDialog() != true) return;
        try
        {
            _export.ExportCsv(_store.Items, dlg.FileName);
            UserMessage.Info("导出完成", "CSV 已导出。");
            LocalDiagnostics.Event("results.export_csv");
        }
        catch (Exception ex)
        {
            UserMessage.Error("导出失败", ex);
        }
    }

    private void SyncChecks()
    {
        foreach (var row in Rows)
        {
            var live = _store.GetAt(row.StoreIndex);
            if (live != null)
                live.IsChecked = row.IsChecked;
        }
    }

    private static string MetaFor(DetectionResult r)
    {
        var text = $"预测：{r.Class}  ·  {r.ConfidenceText}";
        if (!string.IsNullOrEmpty(r.MaxClass) && r.MaxClass != r.Class)
            text += $"（最高：{r.MaxClass} {r.MaxConfidence * 100:0.0}%）";
        return text;
    }

    private void LoadPreview(string path)
    {
        PreviewImage = ThumbnailCache.Shared.GetOrLoad(path, 720);
    }
}
