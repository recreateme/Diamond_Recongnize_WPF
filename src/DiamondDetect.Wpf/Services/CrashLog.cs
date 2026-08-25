using System.IO;
using System.Text;
using System.Windows;
using System.Windows.Threading;

namespace DiamondDetect.Wpf.Services;

/// <summary>默认开启的本地崩溃日志（不上传）。写入 logs/crash_*.txt。</summary>
public static class CrashLog
{
    private static string? _appRoot;

    public static void Install(string appRoot)
    {
        _appRoot = appRoot;
        Application.Current.DispatcherUnhandledException += OnDispatcherUnhandled;
        AppDomain.CurrentDomain.UnhandledException += OnUnhandled;
        TaskScheduler.UnobservedTaskException += OnUnobservedTask;
    }

    private static void OnDispatcherUnhandled(object sender, DispatcherUnhandledExceptionEventArgs e)
    {
        Write("DispatcherUnhandledException", e.Exception);
        // 记录后标记已处理，避免同类 UI 绑定异常直接杀进程；用户仍可通过 crash 日志排查
        e.Handled = true;
    }

    private static void OnUnhandled(object? sender, UnhandledExceptionEventArgs e)
    {
        if (e.ExceptionObject is Exception ex)
            Write("UnhandledException", ex);
        else
            Write("UnhandledException", new Exception(e.ExceptionObject?.ToString() ?? "unknown"));
    }

    private static void OnUnobservedTask(object? sender, UnobservedTaskExceptionEventArgs e)
    {
        Write("UnobservedTaskException", e.Exception);
        e.SetObserved();
    }

    public static string? Write(string source, Exception ex)
    {
        try
        {
            var root = _appRoot ?? AppContext.BaseDirectory;
            var dir = Path.Combine(root, "logs");
            Directory.CreateDirectory(dir);
            var path = Path.Combine(dir, $"crash_{DateTime.Now:yyyyMMdd_HHmmss_fff}.txt");
            var sb = new StringBuilder();
            sb.AppendLine($"time: {DateTime.Now:O}");
            sb.AppendLine($"source: {source}");
            sb.AppendLine($"version: {Core.AppSession.AppVersion}");
            sb.AppendLine($"bridge: {Core.BridgeApi.Version}");
            sb.AppendLine();
            sb.AppendLine(ex.ToString());
            File.WriteAllText(path, sb.ToString(), Encoding.UTF8);
            return path;
        }
        catch
        {
            return null;
        }
    }
}
