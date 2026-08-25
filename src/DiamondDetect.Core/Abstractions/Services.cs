using DiamondDetect.Core.Models;

namespace DiamondDetect.Core.Abstractions;

public interface IInferenceEngine : IDisposable
{
    bool IsLoaded { get; }
    string Backend { get; }
    string Device { get; }
    IReadOnlyList<string> Classes { get; }

    /// <summary>返回人类可读状态消息（对齐原 load 返回值）。</summary>
    string Load(string? ptPath, string? onnxPath, bool useGpu);

    DetectionResult Predict(string imagePath);

    IReadOnlyList<DetectionResult> PredictBatch(
        IReadOnlyList<string> imagePaths,
        IProgress<(int current, int total)>? progress = null,
        CancellationToken cancellationToken = default);
}

public sealed class SahiRunOptions
{
    public string YoloPath { get; init; } = "";
    public string OutputDir { get; init; } = "sahi_output";
    public string Device { get; init; } = "auto";
    public int SliceSize { get; init; } = 1280;
    public double Overlap { get; init; } = 0.20;
    public double DetConf { get; init; } = 0.35;
    public int BatchSize { get; init; } = 8;
    public int CropPadding { get; init; } = 15;
    public double IosThresh { get; init; } = 0.60;
    public double MinAreaRatio { get; init; } = 0.45;
    public double MaxAspectRatio { get; init; } = 1.5;
    public bool EdgeFilter { get; init; } = true;
    public int EdgeMarginPx { get; init; } = 20;
}

public sealed class SahiImageStats
{
    public string Image { get; set; } = "";
    public int TotalDiamonds { get; set; }
    public Dictionary<string, int> DefectCounts { get; set; } = new();
    public double DetectionTimeS { get; set; }
    public double ClassificationTimeS { get; set; }
    public double TotalTimeS { get; set; }
    public string OutputDir { get; set; } = "";
    public string? Error { get; set; }
    public int ContainedSkipped { get; set; }
    public int SmallSkipped { get; set; }
    public int AspectSkipped { get; set; }
    public int EdgeSkipped { get; set; }

    public int SkippedTotal => ContainedSkipped + SmallSkipped + AspectSkipped + EdgeSkipped;

    public string DefectDistribution
    {
        get
        {
            var dist = string.Join(" | ", DefectCounts.Select(kv => $"{kv.Key}:{kv.Value}"));
            if (!string.IsNullOrEmpty(Error))
                return string.IsNullOrEmpty(dist) ? $"错误: {Error}" : dist + $" · 错误: {Error}";
            if (SkippedTotal > 0 && string.IsNullOrEmpty(Error))
                dist = (string.IsNullOrEmpty(dist) ? "" : dist + " · ") + $"后处理剔除 {SkippedTotal}";
            return dist;
        }
    }
}

public sealed class SahiProgress
{
    public int Current { get; init; }
    public int Total { get; init; }
    public string Message { get; init; } = "";
    public SahiImageStats? LastImage { get; init; }
}

public interface ISahiPipeline
{
    Task<IReadOnlyList<SahiImageStats>> ProcessImagesAsync(
        IReadOnlyList<string> imagePaths,
        SahiRunOptions options,
        IProgress<SahiProgress>? progress = null,
        CancellationToken cancellationToken = default);
}

public interface ITrainRunner
{
    Task<int> RunAsync(
        IReadOnlyList<string> arguments,
        IProgress<string>? logLines = null,
        CancellationToken cancellationToken = default);
}

public interface IPythonRuntimeHost : IDisposable
{
    bool IsInitialized { get; }
    string? PythonHome { get; }
    string? PythonDll { get; }
    string PythonCoreDir { get; }
    string Initialize();
}
