using System.Collections.ObjectModel;
using System.IO;
using System.Text.Json;
using System.Windows;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using DiamondDetect.Core;
using DiamondDetect.Core.Abstractions;
using DiamondDetect.Core.Services;
using DiamondDetect.Wpf.Services;
using DiamondDetect.Wpf.Views;

namespace DiamondDetect.Wpf.ViewModels;

public partial class RetrainViewModel : ObservableObject
{
    private readonly AppSession _session;
    private readonly ITrainRunner _train;
    private readonly IInferenceEngine _engine;
    private readonly IConfigService _config;
    private readonly MainViewModel _main;
    private CancellationTokenSource? _cts;

    public RetrainViewModel(
        AppSession session,
        ITrainRunner train,
        IInferenceEngine engine,
        IConfigService config,
        MainViewModel main)
    {
        _session = session;
        _train = train;
        _engine = engine;
        _config = config;
        _main = main;
        IsDeploy = session.IsDeploy;
        ApplyLockState();
    }

    public ObservableCollection<string> LogLines { get; } = new();

    public bool IsDeploy { get; }

    [ObservableProperty] private string lockStatus = "";
    [ObservableProperty] private string unlockButtonText = "输入密码解锁";
    [ObservableProperty] private bool formEnabled;
    [ObservableProperty] private string statsText = "点击「刷新统计」查看数据情况";
    [ObservableProperty] private int imgSize = 128;
    [ObservableProperty] private int epochsPhase1 = 10;
    [ObservableProperty] private int epochsPhase2 = 25;
    [ObservableProperty] private int batchSize = 16;
    [ObservableProperty] private bool isTraining;
    [ObservableProperty] private bool canApply;

    [RelayCommand]
    private void ToggleLock()
    {
        if (_session.AdminUnlocked)
        {
            _session.AdminUnlocked = false;
            if (IsTraining)
                _cts?.Cancel();
            ApplyLockState();
            return;
        }

        var dlg = new PasswordPromptWindow { Owner = Application.Current.MainWindow };
        if (dlg.ShowDialog() == true)
        {
            if (dlg.Password == AppSession.AdminPagePassword)
            {
                _session.AdminUnlocked = true;
                ApplyLockState();
            }
            else
            {
                MessageBox.Show("密码错误。", "管理员解锁", MessageBoxButton.OK, MessageBoxImage.Warning);
            }
        }
    }

    [RelayCommand]
    private void RefreshStats()
    {
        var dataDir = Resolve(_session.Config.DataDir);
        var corrDir = Resolve(_session.Config.CorrectionsDir);
        var lines = new List<string>
        {
            DescribeDir("原始数据 data/", dataDir),
            DescribeDir("修正数据 corrections/", corrDir),
        };
        StatsText = string.Join("\n", lines);

        if (_session.AdminUnlocked || !IsDeploy)
        {
            var cfgPath = Path.Combine(Path.GetDirectoryName(Resolve(_session.Config.PtPath)) ?? "", "train_config.json");
            if (string.IsNullOrEmpty(_session.Config.PtPath))
                cfgPath = Path.Combine(_session.AppRoot, "checkpoints", "train_config.json");
            if (File.Exists(cfgPath))
            {
                try
                {
                    using var doc = JsonDocument.Parse(File.ReadAllText(cfgPath));
                    if (doc.RootElement.TryGetProperty("img_size", out var img))
                        ImgSize = img.GetInt32();
                }
                catch { /* ignore */ }
            }
        }
    }

    [RelayCommand]
    private async Task StartTrainAsync()
    {
        if (IsDeploy && !_session.AdminUnlocked)
        {
            MessageBox.Show("请先输入密码解除只读后再训练。", "只读模式",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        if (!_session.AdminUnlocked && !IsDeploy)
        {
            // 开发版也要求解锁（与原版一致：默认锁定）
            MessageBox.Show("请先输入密码解除只读后再训练。", "只读模式",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        var dataDir = Resolve(_session.Config.DataDir);
        var corrDir = Resolve(_session.Config.CorrectionsDir);
        var pt = Resolve(_session.Config.PtPath);
        var saveDir = string.IsNullOrEmpty(pt)
            ? Path.Combine(_session.AppRoot, "checkpoints")
            : Path.GetDirectoryName(pt)!;

        var args = new List<string>
        {
            "--data_dir", dataDir,
            "--img_size", ImgSize.ToString(),
            "--batch_size", BatchSize.ToString(),
            "--epochs_phase1", EpochsPhase1.ToString(),
            "--epochs_phase2", EpochsPhase2.ToString(),
            "--save_dir", saveDir,
            "--num_workers", "0",
        };
        if (Directory.Exists(corrDir))
        {
            args.Add("--extra_data_dirs");
            args.Add(corrDir);
        }

        LogLines.Clear();
        IsTraining = true;
        _main.IsBusy = true;
        CanApply = false;
        FormEnabled = false;
        _cts = new CancellationTokenSource();
        var progress = new Progress<string>(line =>
        {
            Application.Current.Dispatcher.Invoke(() =>
            {
                LogLines.Add(line);
                if (LogLines.Count > 2000)
                    LogLines.RemoveAt(0);
            });
        });

        try
        {
            using var _ = LocalDiagnostics.Measure("train.run");
            var code = await _train.RunAsync(args, progress, _cts.Token);
            var ok = code == 0;
            LogLines.Add(ok ? "\n[完成] 训练完成 ✓" : $"\n[失败] 训练失败 (code={code})");
            CanApply = ok && _session.AdminUnlocked;
            _main.StatusText = ok ? "训练完成" : "训练失败";
            LocalDiagnostics.Event("train.run.result", $"code={code}");
        }
        catch (OperationCanceledException)
        {
            LogLines.Add("\n[已停止]");
            _main.StatusText = "训练已停止";
            LocalDiagnostics.Event("train.run.cancelled");
        }
        catch (Exception ex)
        {
            LogLines.Add("\n[错误] " + ex.Message);
            LocalDiagnostics.Error("train.run", ex);
            MessageBox.Show(UserMessage.Format(ex), "训练失败", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            IsTraining = false;
            _main.IsBusy = false;
            ApplyLockState();
            _cts?.Dispose();
            _cts = null;
        }
    }

    [RelayCommand]
    private void StopTrain() => _cts?.Cancel();

    [RelayCommand]
    private async Task ApplyModelAsync()
    {
        if (!_session.AdminUnlocked)
        {
            MessageBox.Show("请先输入密码解除只读后再应用模型。", "只读模式",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        try
        {
            var cfg = _config.Current;
            var pt = Resolve(cfg.PtPath);
            var onnx = Resolve(cfg.OnnxPath);
            var msg = await Task.Run(() => _engine.Load(
                File.Exists(pt) ? pt : null,
                File.Exists(onnx) ? onnx : null,
                cfg.UseGpu));
            _main.EngineInfo = $"{_engine.Backend} / {_engine.Device} · 类别 {_engine.Classes.Count}";
            _main.StatusText = msg;
            CanApply = false;
            MessageBox.Show(msg, "模型已更新", MessageBoxButton.OK, MessageBoxImage.Information);
        }
        catch (Exception ex)
        {
            MessageBox.Show(UserMessage.Format(ex), "更新失败", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private void ApplyLockState()
    {
        if (IsDeploy && !_session.AdminUnlocked)
        {
            FormEnabled = false;
            LockStatus = "机台版默认只读 — 输入管理员密码后可训练";
            UnlockButtonText = "输入密码解锁";
        }
        else if (!_session.AdminUnlocked)
        {
            FormEnabled = false;
            LockStatus = "已锁定 — 输入管理员密码后可训练";
            UnlockButtonText = "输入密码解锁";
        }
        else
        {
            FormEnabled = !IsTraining;
            LockStatus = "已解锁（管理员）";
            UnlockButtonText = "重新锁定";
        }
    }

    private string Resolve(string? relative)
    {
        if (string.IsNullOrWhiteSpace(relative)) return "";
        if (Path.IsPathRooted(relative)) return relative;
        return Path.GetFullPath(Path.Combine(_session.AppRoot, relative));
    }

    private static string DescribeDir(string label, string dir)
    {
        if (!Directory.Exists(dir))
            return $"{label}：目录不存在";
        var classCounts = new SortedDictionary<string, int>();
        var total = 0;
        foreach (var clsDir in Directory.GetDirectories(dir))
        {
            var n = ImageFormats.EnumerateImages(clsDir).Count;
            classCounts[Path.GetFileName(clsDir)] = n;
            total += n;
        }
        var clsStr = string.Join("  ", classCounts.Select(kv => $"{kv.Key}:{kv.Value}"));
        return $"  {label}  共 {total} 张\n    {clsStr}";
    }
}
