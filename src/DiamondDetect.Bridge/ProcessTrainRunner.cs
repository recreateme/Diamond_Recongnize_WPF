using System.Diagnostics;
using System.Text;
using DiamondDetect.Core.Abstractions;

namespace DiamondDetect.Bridge;

/// <summary>子进程运行 python_core/train.py，对齐原 TrainWorker。</summary>
public sealed class ProcessTrainRunner : ITrainRunner
{
    private readonly IPythonRuntimeHost _host;
    private Process? _proc;

    public ProcessTrainRunner(IPythonRuntimeHost host)
    {
        _host = host;
    }

    public async Task<int> RunAsync(
        IReadOnlyList<string> arguments,
        IProgress<string>? logLines = null,
        CancellationToken cancellationToken = default)
    {
        _host.Initialize();
        var pythonExe = ResolvePythonExe(_host.PythonHome!);
        var script = Path.Combine(_host.PythonCoreDir, "train.py");
        if (!File.Exists(script))
            throw new FileNotFoundException("训练脚本不存在", script);
        if (!File.Exists(pythonExe))
            throw new FileNotFoundException("未找到 python.exe", pythonExe);

        var argLine = Quote(script) + " " + string.Join(" ", arguments.Select(Quote));
        var psi = new ProcessStartInfo
        {
            FileName = pythonExe,
            Arguments = argLine,
            WorkingDirectory = _host.PythonCoreDir,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };

        var path = Environment.GetEnvironmentVariable("PATH") ?? "";
        var prepend = string.Join(
            Path.PathSeparator,
            PythonRuntimeHost.EnumerateNativeSearchDirs(_host.PythonHome!).Where(Directory.Exists));
        psi.Environment["PATH"] = prepend + Path.PathSeparator + path;
        psi.Environment["PYTHONHOME"] = _host.PythonHome!;
        psi.Environment["PYTHONUTF8"] = "1";
        psi.Environment["CUDA_MODULE_LOADING"] = "LAZY";
        if (string.Equals(Path.GetFileName(_host.PythonHome!.TrimEnd(Path.DirectorySeparatorChar)),
                "python_runtime", StringComparison.OrdinalIgnoreCase))
        {
            psi.Environment.Remove("CUDA_PATH");
            psi.Environment.Remove("CUDA_HOME");
        }

        logLines?.Report($"$ {pythonExe} {argLine}");

        _proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
        var tcs = new TaskCompletionSource<int>();

        _proc.OutputDataReceived += (_, e) =>
        {
            if (e.Data != null) logLines?.Report(e.Data);
        };
        _proc.ErrorDataReceived += (_, e) =>
        {
            if (e.Data != null) logLines?.Report(e.Data);
        };
        _proc.Exited += (_, _) =>
        {
            try { tcs.TrySetResult(_proc.ExitCode); }
            catch (Exception ex) { tcs.TrySetException(ex); }
        };

        if (!_proc.Start())
            throw new InvalidOperationException("无法启动训练进程。");

        _proc.BeginOutputReadLine();
        _proc.BeginErrorReadLine();

        await using var reg = cancellationToken.Register(() =>
        {
            try
            {
                if (!_proc.HasExited)
                    _proc.Kill(entireProcessTree: true);
            }
            catch { /* ignore */ }
        });

        var code = await tcs.Task.ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();
        return code;
    }

    private static string ResolvePythonExe(string home)
    {
        var candidates = new[]
        {
            Path.Combine(home, "python.exe"),
            Path.Combine(home, "python3.exe"),
        };
        return candidates.FirstOrDefault(File.Exists) ?? candidates[0];
    }

    private static string Quote(string value)
    {
        if (string.IsNullOrEmpty(value)) return "\"\"";
        if (value.Contains(' ') || value.Contains('"'))
            return "\"" + value.Replace("\"", "\\\"") + "\"";
        return value;
    }
}
