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

    /// <summary>仅 SAHI 检测定位，不分类、不写 crops/可视化/summary。</summary>
    public bool DetectOnly { get; init; }

    /// <summary>检测前将最长边缩放到 DownsampleMaxSide（仅当源图更大时）。</summary>
    public bool DownsampleEnabled { get; init; }

    public int DownsampleMaxSide { get; init; } = 2560;

    /// <summary>area / linear / cubic / nearest</summary>
    public string DownsampleInterpolation { get; init; } = "area";

    /// <summary>勾选后在输出子目录写入可视化 JPEG（完整模式：分类着色；仅检测：绿框）。</summary>
    public bool SaveVisualization { get; init; }
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
    public bool DetectOnly { get; set; }
    public string? BoxesJsonPath { get; set; }
    public string? BoxesCsvPath { get; set; }

    public int SkippedTotal => ContainedSkipped + SmallSkipped + AspectSkipped + EdgeSkipped;

    public string DefectDistribution
    {
        get
        {
            if (DetectOnly)
            {
                var loc = $"仅定位 · {TotalDiamonds} 颗";
                if (!string.IsNullOrEmpty(Error))
                    return loc + $" · 错误: {Error}";
                if (SkippedTotal > 0)
                    loc += $" · 后处理剔除 {SkippedTotal}";
                return loc;
            }
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

public sealed class UniformityScoreRow
{
    public string Image { get; set; } = "";
    public string SourceJson { get; set; } = "";
    public string ScoresJson { get; set; } = "";
    public string VisPath { get; set; } = "";
    public string SourceKind { get; set; } = "";
    public int NPoints { get; set; }
    public double? VoronoiAreaCvNormalized { get; set; }
    public double? NnDistanceCvNormalized { get; set; }
    public double? ClarkEvansR { get; set; }
    public double? DelaunayEdgeCvNormalized { get; set; }
    public double? GridDensityCv { get; set; }
    public string Status { get; set; } = "";
    public string ConfFilter { get; set; } = "";
}

public sealed class UniformityTileInfo
{
    public string Image { get; set; } = "";
    public string JsonPath { get; set; } = "";
    public string SourceKind { get; set; } = "";
}

public sealed class UniformityBatchResult
{
    public string OutputRoot { get; set; } = "";
    public string? SummaryCsv { get; set; }
    public IReadOnlyList<UniformityScoreRow> Rows { get; set; } = Array.Empty<UniformityScoreRow>();
}

public sealed class UniformityVisResult
{
    public string VisPath { get; set; } = "";
    public string Status { get; set; } = "";
    public string Background { get; set; } = "";
    public int NPoints { get; set; }
}

public interface IUniformityAnalyzer
{
    /// <summary>列出产品输出根下可分析的 tile（优先 detect_boxes.json，否则 result.json）。</summary>
    Task<IReadOnlyList<UniformityTileInfo>> ListTilesAsync(
        string outputRoot,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// 分析输出根；jsonPaths 为空则全部 tile，否则仅分析勾选子集。
    /// 写回各子目录 uniformity_scores.json 与根目录 uniformity_summary.csv。
    /// </summary>
    Task<UniformityBatchResult> AnalyzeOutputRootAsync(
        string outputRoot,
        double confThreshold,
        IReadOnlyList<string>? jsonPaths = null,
        bool writeVisualization = false,
        IProgress<(int current, int total, string message)>? progress = null,
        CancellationToken cancellationToken = default);

    /// <summary>按需生成单张 uniformity_vis.jpg。</summary>
    Task<UniformityVisResult> RenderVisualizationAsync(
        string jsonPath,
        double confThreshold,
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
