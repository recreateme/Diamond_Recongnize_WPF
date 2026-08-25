using System.Text;
using System.Windows;

namespace DiamondDetect.Wpf.Services;

/// <summary>面向操作员的错误文案整理（展开内部异常、附加常见排查提示）。</summary>
public static class UserMessage
{
    public static string Format(Exception ex)
    {
        ex = Unwrap(ex);
        var sb = new StringBuilder(ex.Message?.Trim() ?? "未知错误");

        var hint = HintFor(ex);
        if (!string.IsNullOrEmpty(hint))
        {
            sb.AppendLine();
            sb.AppendLine();
            sb.Append(hint);
        }
        return sb.ToString();
    }

    public static void Error(string title, Exception ex)
    {
        LocalDiagnostics.Error("ui." + title, ex);
        MessageBox.Show(Format(ex), title, MessageBoxButton.OK, MessageBoxImage.Error);
    }

    public static void Warn(string title, string message)
        => MessageBox.Show(message, title, MessageBoxButton.OK, MessageBoxImage.Warning);

    public static void Info(string title, string message)
        => MessageBox.Show(message, title, MessageBoxButton.OK, MessageBoxImage.Information);

    private static Exception Unwrap(Exception ex)
    {
        while (true)
        {
            if (ex is AggregateException { InnerExceptions.Count: 1 } ae)
            {
                ex = ae.InnerExceptions[0];
                continue;
            }
            if (ex.InnerException is { } inner
                && (ex is AggregateException || ex.GetType().Name.Contains("TargetInvocation", StringComparison.Ordinal)))
            {
                ex = inner;
                continue;
            }
            break;
        }
        return ex;
    }

    private static string? HintFor(Exception ex)
    {
        var text = (ex.Message ?? "") + " " + ex.GetType().Name;
        if (ContainsAny(text, "python", "Python.Runtime", "DLL", "python312", "DIAMOND_PYTHON"))
            return "排查：确认环境变量 DIAMOND_PYTHON_HOME / DIAMOND_PYTHON_DLL，以及程序目录下存在 python_core。";
        if (ContainsAny(text, "CUDA", "cuda", "onnxruntime", "ORT", "cublas"))
            return "排查：GPU/驱动或 ONNX Runtime 异常时可在「设置」关闭「使用 GPU」后重试。";
        if (ContainsAny(text, "model", "checkpoint", "onnx", "weights", "best.pt", "YOLO"))
            return "排查：在「设置」中检查分类模型 / YOLO 权重路径是否存在，必要时重新加载模型。";
        if (ContainsAny(text, "Unauthorized", "Access", "拒绝访问", "IOException"))
            return "排查：确认目标路径可写，文件未被其他程序占用。";
        return null;
    }

    private static bool ContainsAny(string haystack, params string[] needles)
    {
        foreach (var n in needles)
        {
            if (haystack.Contains(n, StringComparison.OrdinalIgnoreCase))
                return true;
        }
        return false;
    }
}
