using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using System.Windows.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using DiamondDetect.Core;
using DiamondDetect.Core.Abstractions;
using DiamondDetect.Core.Models;
using DiamondDetect.Core.Services;
using DiamondDetect.Wpf.Services;
using Microsoft.Win32;

namespace DiamondDetect.Wpf.ViewModels;

public partial class CorrectionListItem : ObservableObject
{
    public int StoreIndex { get; init; }
    public DetectionResult Result { get; init; } = null!;
    public string Title { get; init; } = "";
    public string Subtitle { get; init; } = "";
}

public partial class CorrectionViewModel : ObservableObject
{
    private readonly AppSession _session;
    private readonly ResultStore _store;
    private readonly ExportService _export;
    private readonly IInferenceEngine _engine;
    private readonly MainViewModel _main;

    public CorrectionViewModel(
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
    }

    public ObservableCollection<CorrectionListItem> PendingItems { get; } = new();
    public ObservableCollection<string> ClassNames { get; } = new();

    [ObservableProperty] private string countText = "待修正：0 项";
    [ObservableProperty] private string progressText = "";
    [ObservableProperty] private string fileNameText = "";
    [ObservableProperty] private string predText = "";
    [ObservableProperty] private BitmapImage? currentImage;
    [ObservableProperty] private CorrectionListItem? selectedItem;
    [ObservableProperty] private bool hasPending;
    [ObservableProperty] private string emptyHint =
        "暂无待修正图像\n\n· 在「结果管理」中点击「送修正」\n· 或本页点击「选择文件夹」导入待打标签图像";

    partial void OnSelectedItemChanged(CorrectionListItem? value) => ShowCurrent(value);

    public void Refresh()
    {
        ClassNames.Clear();
        foreach (var c in _engine.Classes)
            ClassNames.Add(c);

        var pending = _store.GetPendingCorrections();
        var prevPath = SelectedItem?.Result.Path;

        PendingItems.Clear();
        foreach (var (idx, r) in pending)
        {
            var sub = r.FromFolderImport || r.Class == "待标注"
                ? (r.Class != "待标注" && !string.IsNullOrEmpty(r.Class)
                    ? $"来源：文件夹导入 · 目录提示：{r.Class}"
                    : "来源：文件夹导入（待重新打标签）")
                : $"预测：{r.Class} ({r.ConfidenceText})";
            PendingItems.Add(new CorrectionListItem
            {
                StoreIndex = idx,
                Result = r,
                Title = r.FileName,
                Subtitle = sub,
            });
        }

        CountText = $"待修正：{PendingItems.Count} 项";
        HasPending = PendingItems.Count > 0;
        _main.UpdateCorrectionBadge(PendingItems.Count);

        if (!HasPending)
        {
            SelectedItem = null;
            ShowEmpty();
            return;
        }

        var match = PendingItems.FirstOrDefault(i =>
            string.Equals(i.Result.Path, prevPath, StringComparison.OrdinalIgnoreCase));
        SelectedItem = match ?? PendingItems[0];
    }

    [RelayCommand]
    private void ImportFolder()
    {
        var dlg = new OpenFolderDialog { Title = "选择待重新打标签的图像文件夹" };
        if (dlg.ShowDialog() != true) return;

        var imgs = ImageFormats.EnumerateImages(dlg.FolderName);
        if (imgs.Count == 0)
        {
            MessageBox.Show($"未找到图像：\n{dlg.FolderName}", "提示",
                MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }

        var known = _engine.Classes.ToHashSet(StringComparer.Ordinal);
        var existing = _store.GetPendingCorrections()
            .Select(x => x.Result.Path)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        var added = 0;
        var skipped = 0;
        foreach (var path in imgs)
        {
            if (existing.Contains(path))
            {
                skipped++;
                continue;
            }
            var parent = Path.GetFileName(Path.GetDirectoryName(path) ?? "");
            var hint = known.Contains(parent) ? parent : "待标注";
            _store.UpsertForceFlag(new DetectionResult
            {
                Path = path,
                Class = hint,
                MaxClass = hint,
                Confidence = 0,
                MaxConfidence = 0,
                FromFolderImport = true,
                Flagged = true,
            });
            existing.Add(path);
            added++;
        }

        Refresh();
        var msg = $"已导入 {added} 张待标注图像" + (skipped > 0 ? $"（跳过重复 {skipped}）" : "");
        _main.StatusText = msg;
        MessageBox.Show(msg + $"\n来源：{dlg.FolderName}", "导入完成",
            MessageBoxButton.OK, MessageBoxImage.Information);
    }

    [RelayCommand]
    private void ApplyClass(string? trueClass)
    {
        if (string.IsNullOrWhiteSpace(trueClass) || SelectedItem is null) return;
        var idx = SelectedItem.StoreIndex;
        var r = _store.GetAt(idx);
        if (r is null) return;
        if (!File.Exists(r.Path))
        {
            MessageBox.Show($"找不到图像：\n{r.Path}", "文件不存在",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        var corrRoot = Path.Combine(_session.AppRoot, _session.Config.CorrectionsDir);
        var saved = _export.ArchiveToCorrections(_store, new[] { idx }, corrRoot, trueClass);
        if (saved == 0)
        {
            MessageBox.Show("无法复制图像到 corrections 目录。", "保存失败",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        _main.StatusText = $"已归档「{r.FileName}」→ {_session.Config.CorrectionsDir}/{trueClass}/";
        Refresh();
    }

    private void ShowCurrent(CorrectionListItem? item)
    {
        if (item is null)
        {
            ShowEmpty();
            return;
        }

        var r = item.Result;
        FileNameText = r.FileName;
        PredText = item.Subtitle;
        var pos = PendingItems.IndexOf(item);
        ProgressText = pos >= 0 ? $"第 {pos + 1} / {PendingItems.Count} 张" : "";
        CurrentImage = ThumbnailCache.Shared.GetOrLoad(r.Path, 900);
    }

    private void ShowEmpty()
    {
        FileNameText = "";
        PredText = "";
        ProgressText = "";
        CurrentImage = null;
    }
}
