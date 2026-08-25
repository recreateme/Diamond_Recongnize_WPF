using System.Runtime.InteropServices;
using DiamondDetect.Core;
using DiamondDetect.Core.Abstractions;
using Python.Runtime;

namespace DiamondDetect.Bridge;

/// <summary>
/// 定位 Conda/嵌入式 Python，初始化 pythonnet，并把 python_core 加入 sys.path。
/// 机台完整包优先使用 exe 同级 python_runtime，并前置 torch/CUDA 原生 DLL 目录。
/// </summary>
public sealed class PythonRuntimeHost : IPythonRuntimeHost
{
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr AddDllDirectory(string newDirectory);

    private readonly object _gate = new();
    private bool _initialized;

    public PythonRuntimeHost(string appRoot, string? pythonHomeOverride = null)
    {
        AppRoot = appRoot;
        PythonCoreDir = Path.GetFullPath(Path.Combine(appRoot, "python_core"));
        PythonHomeOverride = pythonHomeOverride;
    }

    public string AppRoot { get; }
    public string PythonCoreDir { get; }
    public string? PythonHomeOverride { get; }
    public bool IsInitialized => _initialized;
    public string? PythonHome { get; private set; }
    public string? PythonDll { get; private set; }

    public string Initialize()
    {
        lock (_gate)
        {
            if (_initialized)
                return $"Python 已初始化: {PythonHome}";

            if (!Directory.Exists(PythonCoreDir))
                throw new DirectoryNotFoundException($"未找到 python_core: {PythonCoreDir}");

            var (home, dll) = ResolvePython(PythonHomeOverride, AppRoot);
            PythonHome = home;
            PythonDll = dll;

            Environment.SetEnvironmentVariable("PYTHONHOME", home);
            Environment.SetEnvironmentVariable("CUDA_MODULE_LOADING", "LAZY");
            if (IsBundledRuntime(home, AppRoot))
            {
                Environment.SetEnvironmentVariable("DEFECTS_DEPLOY", "1");
                Environment.SetEnvironmentVariable("CUDA_PATH", null);
                Environment.SetEnvironmentVariable("CUDA_HOME", null);
            }

            PrependNativeSearchPaths(home);

            Runtime.PythonDLL = dll;
            if (!PythonEngine.IsInitialized)
            {
                PythonEngine.Initialize();
                PythonEngine.BeginAllowThreads();
            }

            using (Py.GIL())
            {
                dynamic sys = Py.Import("sys");
                var core = PythonCoreDir.Replace('\\', '/');
                bool found = false;
                foreach (var p in sys.path)
                {
                    if (string.Equals(p.ToString(), core, StringComparison.OrdinalIgnoreCase) ||
                        string.Equals(p.ToString(), PythonCoreDir, StringComparison.OrdinalIgnoreCase))
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                    sys.path.insert(0, PythonCoreDir);

                try
                {
                    dynamic appPaths = Py.Import("app_paths");
                    appPaths.setup_ort_dll_paths();
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine("setup_ort_dll_paths: " + ex.Message);
                }
            }

            _initialized = true;
            return $"Python 已初始化\nHOME={home}\nDLL={dll}\npython_core={PythonCoreDir}";
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (!_initialized)
                return;
            try
            {
                if (PythonEngine.IsInitialized)
                    PythonEngine.Shutdown();
            }
            catch
            {
                // ignore shutdown races
            }
            _initialized = false;
        }
    }

    public static (string home, string dll) ResolvePython(string? homeOverride = null, string? appRoot = null)
    {
        var dllEnv = Environment.GetEnvironmentVariable("DIAMOND_PYTHON_DLL");
        var explicitHome = homeOverride
            ?? Environment.GetEnvironmentVariable("DIAMOND_PYTHON_HOME");
        var condaPrefix = Environment.GetEnvironmentVariable("CONDA_PREFIX");

        if (!string.IsNullOrWhiteSpace(dllEnv) && File.Exists(dllEnv))
        {
            var home = explicitHome;
            if (string.IsNullOrWhiteSpace(home))
                home = Path.GetDirectoryName(dllEnv)!;
            return (Path.GetFullPath(home), Path.GetFullPath(dllEnv));
        }

        var candidates = new List<string>();
        // 显式指定 > 包内 python_runtime > 系统 Conda（避免机台误用本机残留 CONDA_PREFIX）
        if (!string.IsNullOrWhiteSpace(explicitHome))
            candidates.Add(explicitHome);
        if (!string.IsNullOrWhiteSpace(appRoot))
            candidates.Add(Path.Combine(appRoot, PackLayout.PythonRuntimeDir));
        if (!string.IsNullOrWhiteSpace(condaPrefix))
            candidates.Add(condaPrefix);

        var user = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        candidates.AddRange(new[]
        {
            Path.Combine(user, "anaconda3", "envs", "cv-yolo"),
            Path.Combine(user, "miniconda3", "envs", "cv-yolo"),
            Path.Combine(user, "mambaforge", "envs", "cv-yolo"),
            Path.Combine(user, "miniforge3", "envs", "cv-yolo"),
            @"D:\Software\MiniAnaconda\envs\cv-yolo",
            @"C:\ProgramData\miniconda3\envs\cv-yolo",
            @"C:\ProgramData\anaconda3\envs\cv-yolo",
            @"C:\Users\Public\miniconda3\envs\cv-yolo",
        });

        foreach (var home in candidates.Where(Directory.Exists).Select(Path.GetFullPath).Distinct())
        {
            var dll = FindPythonDll(home);
            if (dll != null)
                return (home, dll);
        }

        throw new InvalidOperationException(
            "未能定位 Python。请将完整包中的 python_runtime 与 exe 放在同级，或设置：\n" +
            "  DIAMOND_PYTHON_HOME = conda 环境根目录\n" +
            "  DIAMOND_PYTHON_DLL  = python3xx.dll 完整路径");
    }

    internal static IEnumerable<string> EnumerateNativeSearchDirs(string home)
    {
        yield return home;
        yield return Path.Combine(home, "Scripts");
        yield return Path.Combine(home, "DLLs");
        yield return Path.Combine(home, "Library", "bin");

        var torchLib = Path.Combine(home, "Lib", "site-packages", "torch", "lib");
        if (Directory.Exists(torchLib))
            yield return torchLib;

        var nvidiaRoot = Path.Combine(home, "Lib", "site-packages", "nvidia");
        if (Directory.Exists(nvidiaRoot))
        {
            foreach (var pkg in Directory.GetDirectories(nvidiaRoot))
            {
                if (string.Equals(Path.GetFileName(pkg), "cudnn", StringComparison.OrdinalIgnoreCase))
                    continue;
                var bin = Path.Combine(pkg, "bin");
                if (Directory.Exists(bin))
                    yield return bin;
            }
        }
    }

    private static void PrependNativeSearchPaths(string home)
    {
        var dirs = EnumerateNativeSearchDirs(home)
            .Where(Directory.Exists)
            .Select(Path.GetFullPath)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();

        foreach (var dir in dirs)
        {
            try { AddDllDirectory(dir); }
            catch { /* older OS or invalid path */ }
        }

        var path = Environment.GetEnvironmentVariable("PATH") ?? "";
        Environment.SetEnvironmentVariable("PATH", string.Join(Path.PathSeparator, dirs) + Path.PathSeparator + path);
    }

    private static bool IsBundledRuntime(string home, string appRoot)
    {
        var bundled = Path.Combine(appRoot, "python_runtime");
        return home.StartsWith(bundled, StringComparison.OrdinalIgnoreCase)
               || string.Equals(Path.GetFileName(home.TrimEnd(Path.DirectorySeparatorChar)), "python_runtime",
                   StringComparison.OrdinalIgnoreCase);
    }

    private static string? FindPythonDll(string home)
    {
        foreach (var dll in Directory.GetFiles(home, "python3*.dll"))
        {
            var name = Path.GetFileNameWithoutExtension(dll).ToLowerInvariant();
            if (name is "python3" or "python313" or "python312" or "python311" or "python310" or "python39")
            {
                if (name != "python3")
                    return dll;
            }
        }

        foreach (var dll in Directory.GetFiles(home, "python3*.dll"))
        {
            var name = Path.GetFileNameWithoutExtension(dll);
            if (!string.Equals(name, "python3", StringComparison.OrdinalIgnoreCase))
                return dll;
        }

        var fallback = Path.Combine(home, "python3.dll");
        return File.Exists(fallback) ? fallback : null;
    }
}
