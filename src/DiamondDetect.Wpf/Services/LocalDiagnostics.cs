using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.Json;

namespace DiamondDetect.Wpf.Services;

/// <summary>
/// 本地诊断日志（默认关闭，不上传）。开启后写入 logs/diag.jsonl。
/// </summary>
public static class LocalDiagnostics
{
    private static readonly object Gate = new();
    private static string? _logPath;
    private static bool _enabled;

    public static bool Enabled
    {
        get { lock (Gate) return _enabled; }
        set
        {
            lock (Gate)
            {
                _enabled = value;
            }
        }
    }

    public static void Configure(string appRoot, bool enabled)
    {
        lock (Gate)
        {
            _enabled = enabled;
            var dir = Path.Combine(appRoot, "logs");
            Directory.CreateDirectory(dir);
            _logPath = Path.Combine(dir, "diag.jsonl");
        }

        if (enabled)
            Event("diag.enabled", $"bridge={Core.BridgeApi.Version}");
    }

    public static void Event(string name, string? detail = null)
        => Write("event", name, detail, null);

    public static void Error(string name, Exception ex)
        => Write("error", name, UserMessage.Format(ex).Split('\n')[0], ex.GetType().Name);

    public static IDisposable Measure(string name, string? detail = null)
        => new Scope(name, detail);

    private static void Write(string kind, string name, string? detail, string? errorType)
    {
        string? path;
        lock (Gate)
        {
            if (!_enabled || string.IsNullOrEmpty(_logPath))
                return;
            path = _logPath;
        }

        try
        {
            var payload = new Dictionary<string, object?>
            {
                ["ts"] = DateTime.Now.ToString("O"),
                ["kind"] = kind,
                ["name"] = name,
            };
            if (!string.IsNullOrEmpty(detail))
                payload["detail"] = detail;
            if (!string.IsNullOrEmpty(errorType))
                payload["error_type"] = errorType;

            var line = JsonSerializer.Serialize(payload) + Environment.NewLine;
            lock (Gate)
            {
                File.AppendAllText(path!, line, Encoding.UTF8);
            }
        }
        catch
        {
            // never break app for diagnostics
        }
    }

    private sealed class Scope : IDisposable
    {
        private readonly string _name;
        private readonly string? _detail;
        private readonly Stopwatch _sw = Stopwatch.StartNew();
        private bool _done;

        public Scope(string name, string? detail)
        {
            _name = name;
            _detail = detail;
            Event(_name + ".start", _detail);
        }

        public void Dispose()
        {
            if (_done) return;
            _done = true;
            _sw.Stop();
            var d = string.IsNullOrEmpty(_detail)
                ? $"ms={_sw.ElapsedMilliseconds}"
                : $"{_detail};ms={_sw.ElapsedMilliseconds}";
            Event(_name + ".end", d);
        }
    }
}
