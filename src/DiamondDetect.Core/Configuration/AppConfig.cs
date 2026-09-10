using System.Text.Json.Serialization;

namespace DiamondDetect.Core.Configuration;

/// <summary>
/// 与源项目 app_config.json 字段兼容的应用配置。
/// </summary>
public sealed class AppConfig
{
    [JsonPropertyName("pt_path")]
    public string PtPath { get; set; } = "checkpoints/best_model.pt";

    [JsonPropertyName("onnx_path")]
    public string OnnxPath { get; set; } = "checkpoints/model.onnx";

    [JsonPropertyName("data_dir")]
    public string DataDir { get; set; } = "data";

    [JsonPropertyName("corrections_dir")]
    public string CorrectionsDir { get; set; } = "corrections";

    [JsonPropertyName("use_gpu")]
    public bool UseGpu { get; set; } = true;

    /// <summary>本地诊断日志（logs/diag.jsonl），默认关闭，不上传。</summary>
    [JsonPropertyName("enable_local_diagnostics")]
    public bool EnableLocalDiagnostics { get; set; } = false;

    [JsonPropertyName("yolo_path")]
    public string YoloPath { get; set; } = "";

    [JsonPropertyName("sahi_device")]
    public string SahiDevice { get; set; } = "auto";

    [JsonPropertyName("sahi_slice_size")]
    public int SahiSliceSize { get; set; } = 1280;

    [JsonPropertyName("sahi_overlap")]
    public double SahiOverlap { get; set; } = 0.20;

    [JsonPropertyName("sahi_det_conf")]
    public double SahiDetConf { get; set; } = 0.35;

    [JsonPropertyName("sahi_batch_size")]
    public int SahiBatchSize { get; set; } = 8;

    [JsonPropertyName("sahi_crop_padding")]
    public int SahiCropPadding { get; set; } = 15;

    [JsonPropertyName("sahi_output_dir")]
    public string SahiOutputDir { get; set; } = "sahi_output";

    [JsonPropertyName("sahi_ios_thresh")]
    public double SahiIosThresh { get; set; } = 0.60;

    [JsonPropertyName("sahi_min_area_ratio")]
    public double SahiMinAreaRatio { get; set; } = 0.45;

    [JsonPropertyName("sahi_max_aspect_ratio")]
    public double SahiMaxAspectRatio { get; set; } = 1.5;

    [JsonPropertyName("sahi_edge_filter")]
    public bool SahiEdgeFilter { get; set; } = true;

    [JsonPropertyName("sahi_edge_margin_px")]
    public int SahiEdgeMarginPx { get; set; } = 20;

    /// <summary>均匀度分析：YOLO det_conf 过滤阈值（默认 0.25）。</summary>
    [JsonPropertyName("uniformity_conf_threshold")]
    public double UniformityConfThreshold { get; set; } = 0.25;

    public static AppConfig CreateDefault(bool deployMode)
    {
        var cfg = new AppConfig();
        if (deployMode)
        {
            cfg.PtPath = "";
            cfg.YoloPath = "detect_weights/best.pt";
        }
        return cfg;
    }
}
