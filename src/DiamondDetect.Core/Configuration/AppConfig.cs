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
    public int SahiSliceSize { get; set; } = 1248;

    [JsonPropertyName("sahi_overlap")]
    public double SahiOverlap { get; set; } = 0.2244;

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

    /// <summary>检测页输出选项：保存可视化（默认关，跨会话记住）。</summary>
    [JsonPropertyName("ui_save_visualization")]
    public bool UiSaveVisualization { get; set; }

    /// <summary>检测页输出选项：保存 crop/（默认关）。</summary>
    [JsonPropertyName("ui_save_crops")]
    public bool UiSaveCrops { get; set; }

    /// <summary>检测页输出选项：写出 detect_boxes.json/csv（默认关）。</summary>
    [JsonPropertyName("ui_save_boxes")]
    public bool UiSaveBoxes { get; set; }

    /// <summary>检测页输出选项：跑完后计算均匀度（默认关；需完整模式且已勾位置文件）。</summary>
    [JsonPropertyName("ui_run_uniformity")]
    public bool UiRunUniformity { get; set; }

    /// <summary>检测页：整盘批量时跳过盘外空白图（默认开，跨会话记住）。</summary>
    [JsonPropertyName("ui_skip_empty_tiles")]
    public bool UiSkipEmptyTiles { get; set; } = true;

    /// <summary>整盘 tile 网格行数（编号按行优先）。</summary>
    [JsonPropertyName("tile_grid_rows")]
    public int TileGridRows { get; set; } = 24;

    /// <summary>整盘 tile 网格列数。</summary>
    [JsonPropertyName("tile_grid_cols")]
    public int TileGridCols { get; set; } = 24;

    /// <summary>跳过空白图后的随机抽检回查张数。</summary>
    [JsonPropertyName("skip_audit_count")]
    public int SkipAuditCount { get; set; } = 3;

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
