using System.IO;
using System.Runtime.InteropServices;
using System.Windows;
using DiamondDetect.Bridge;
using DiamondDetect.Core;
using DiamondDetect.Core.Abstractions;
using DiamondDetect.Core.Services;
using DiamondDetect.Wpf.Services;
using DiamondDetect.Wpf.ViewModels;
using Microsoft.Extensions.DependencyInjection;

namespace DiamondDetect.Wpf;

public partial class App : Application
{
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AllocConsole();

    private ServiceProvider? _services;

    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        var appRoot = ResolveAppRoot();
        Directory.SetCurrentDirectory(appRoot);
        ApplyBundledRuntimeEnv(appRoot);
        CrashLog.Install(appRoot);

        if (e.Args.Any(a => string.Equals(a, "--verify", StringComparison.OrdinalIgnoreCase))
            || string.Equals(Environment.GetEnvironmentVariable("DEFECTS_VERIFY"), "1", StringComparison.Ordinal))
        {
            AllocConsole();
            var code = RunVerify(appRoot);
            Shutdown(code);
            return;
        }

        var deployMode = DeployModeDetector.Detect(appRoot);
        var services = new ServiceCollection();
        services.AddSingleton(_ => new ConfigService(appRoot, deployMode.IsDeploy()));
        services.AddSingleton<IConfigService>(sp => sp.GetRequiredService<ConfigService>());
        services.AddSingleton<ResultStore>();
        services.AddSingleton(sp => new AppSession(
            deployMode,
            sp.GetRequiredService<IConfigService>(),
            sp.GetRequiredService<ResultStore>(),
            appRoot));
        services.AddSingleton<IPythonRuntimeHost>(_ => new PythonRuntimeHost(appRoot));
        services.AddSingleton<IInferenceEngine>(sp =>
            new PythonInferenceEngine(sp.GetRequiredService<IPythonRuntimeHost>(), deployMode.IsDeploy()));
        services.AddSingleton<ISahiPipeline>(sp =>
            new PythonSahiPipeline(
                sp.GetRequiredService<IPythonRuntimeHost>(),
                sp.GetRequiredService<IInferenceEngine>()));
        services.AddSingleton<ExportService>();
        services.AddSingleton<ITrainRunner>(sp =>
            new ProcessTrainRunner(sp.GetRequiredService<IPythonRuntimeHost>()));
        services.AddSingleton<MainViewModel>();
        services.AddSingleton<DiamondDetectViewModel>();
        services.AddSingleton<DetectionViewModel>();
        services.AddSingleton<ResultsViewModel>();
        services.AddSingleton<CorrectionViewModel>();
        services.AddSingleton<RetrainViewModel>();
        services.AddSingleton<MainWindow>();

        _services = services.BuildServiceProvider();

        var session = _services.GetRequiredService<AppSession>();
        LocalDiagnostics.Configure(appRoot, session.Config.EnableLocalDiagnostics);

        var window = _services.GetRequiredService<MainWindow>();
        window.Show();
    }

    protected override void OnExit(ExitEventArgs e)
    {
        try
        {
            _services?.GetService<IInferenceEngine>()?.Dispose();
            _services?.GetService<IPythonRuntimeHost>()?.Dispose();
        }
        catch
        {
            // ignore
        }
        _services?.Dispose();
        base.OnExit(e);
    }

    private static string ResolveAppRoot()
    {
        var baseDir = AppContext.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);

        // 机台包：exe 同级即有 python_core / checkpoints
        if (Directory.Exists(Path.Combine(baseDir, "python_core")))
            return baseDir;

        // 开发：从 bin/Debug/... 上溯到含 DiamondDetect.sln 的仓库根
        var dir = new DirectoryInfo(baseDir);
        while (dir != null)
        {
            var pythonCore = Path.Combine(dir.FullName, "python_core");
            var sln = Path.Combine(dir.FullName, "DiamondDetect.sln");
            if (Directory.Exists(pythonCore) && File.Exists(sln))
                return dir.FullName;
            dir = dir.Parent;
        }

        return baseDir;
    }

    /// <summary>双击 exe 时也使用包内 Python，不必依赖启动 bat。</summary>
    private static void ApplyBundledRuntimeEnv(string appRoot)
    {
        var runtime = PackLayout.PythonRuntime(appRoot);
        var dll = Path.Combine(runtime, "python312.dll");
        if (!Directory.Exists(runtime) || !File.Exists(dll))
            return;

        Environment.SetEnvironmentVariable("DEFECTS_DEPLOY", "1");
        Environment.SetEnvironmentVariable("DIAMOND_PYTHON_HOME", runtime);
        Environment.SetEnvironmentVariable("DIAMOND_PYTHON_DLL", dll);
        Environment.SetEnvironmentVariable("PYTHONHOME", runtime);
        Environment.SetEnvironmentVariable("CUDA_MODULE_LOADING", "LAZY");
        Environment.SetEnvironmentVariable("CUDA_PATH", null);
        Environment.SetEnvironmentVariable("CUDA_HOME", null);

        var prepend = string.Join(Path.PathSeparator, new[]
        {
            runtime,
            Path.Combine(runtime, "Scripts"),
            Path.Combine(runtime, "Library", "bin"),
            Path.Combine(runtime, "Lib", "site-packages", "torch", "lib"),
        }.Where(Directory.Exists));
        var path = Environment.GetEnvironmentVariable("PATH") ?? "";
        Environment.SetEnvironmentVariable("PATH", prepend + Path.PathSeparator + path);
    }

    private static int RunVerify(string appRoot)
    {
        try
        {
            Console.WriteLine("=== DEFECTS_VERIFY (WPF) ===");
            using var host = new PythonRuntimeHost(appRoot);
            Console.WriteLine(host.Initialize());
            using var engine = new PythonInferenceEngine(host, deployMode: true);
            var onnx = Path.Combine(appRoot, "checkpoints", "model.onnx");
            var msg = engine.Load(null, onnx, useGpu: true);
            Console.WriteLine(msg);
            Console.WriteLine("device: " + engine.Device);
            if (!string.Equals(engine.Device, "cuda", StringComparison.OrdinalIgnoreCase))
                Console.WriteLine("[INFO] 当前为 ONNX/CPU。有 NVIDIA 独显时将自动使用 CUDA 快速模式。");
            else
                Console.WriteLine("[OK] ONNX CUDA 快速模式");
            return engine.IsLoaded ? 0 : 1;
        }
        catch (Exception ex)
        {
            Console.WriteLine("VERIFY FAIL: " + ex);
            return 1;
        }
    }
}
