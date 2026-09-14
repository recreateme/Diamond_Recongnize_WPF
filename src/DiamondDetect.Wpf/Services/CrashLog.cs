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
        var path = Write("DispatcherUnhandledException", e.Exception);
        // 记录后标记已处理，避免同类 UI 绑定异常直接杀进程；用户仍可通过 crash 日志排查
        e.Handled = true;

        // 【修复】之前这里只写日志、没有任何面向用户的可见提示——操作员点了
        // 按钮，背后触发了一个被这里接住的异常，界面表现为"点了好像没反应"，
        // 只有回头翻 logs/crash_*.txt 才能发现出过错。加一条轻量提示，确保
        // "这里出问题了"这件事至少是操作员能感知到的，而不是完全无声。
        try
        {
            UserMessage.Warn(
                "发生了一个内部错误",
                "刚才的操作未能正常完成，已自动记录到日志"
                + (path != null ? $"：\n{path}" : "。")
                + "\n\n可以重试一次；如果反复出现，请把这份日志发给技术支持。");
        }
        catch
        {
            // 提示本身失败也不应该影响异常已被处理这件事
        }
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
