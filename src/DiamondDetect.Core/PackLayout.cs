namespace DiamondDetect.Core;

/// <summary>机台包约定目录（exe 同级），启动时自动解析，无需进设置页。</summary>
public static class PackLayout
{
    public const string PythonRuntimeDir = "python_runtime";
    public const string PythonCoreDir = "python_core";
    public const string OnnxRelative = "checkpoints/model.onnx";
    public const string YoloRelative = "detect_weights/best.pt";

    public static bool LooksLikeMachinePack(string appRoot)
    {
        if (string.IsNullOrWhiteSpace(appRoot))
            return false;
        if (File.Exists(Path.Combine(appRoot, "DiamondDetect.sln")))
            return false;
        if (Directory.Exists(Path.Combine(appRoot, PythonRuntimeDir)))
            return true;
        return Directory.Exists(Path.Combine(appRoot, PythonCoreDir))
               && File.Exists(Path.Combine(appRoot, "checkpoints", "model.onnx"));
    }

    public static string PythonRuntime(string appRoot)
        => Path.Combine(appRoot, PythonRuntimeDir);

    public static string ClassifyOnnx(string appRoot)
        => Path.Combine(appRoot, "checkpoints", "model.onnx");

    public static string YoloWeights(string appRoot)
        => Path.Combine(appRoot, "detect_weights", "best.pt");

    public static string ResolveFile(string appRoot, string? configured, params string[] fallbacks)
    {
        foreach (var raw in new[] { configured }.Concat(fallbacks))
        {
            if (string.IsNullOrWhiteSpace(raw))
                continue;
            var full = Path.IsPathRooted(raw)
                ? raw
                : Path.GetFullPath(Path.Combine(appRoot, raw));
            if (File.Exists(full))
                return full;
        }
        return "";
    }

    public static string ResolveYolo(string appRoot, string? configured)
        => ResolveFile(appRoot, configured, YoloRelative, "best.pt");

    public static string ResolveOnnx(string appRoot, string? configured)
        => ResolveFile(appRoot, configured, OnnxRelative);
}
