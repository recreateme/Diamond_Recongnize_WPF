namespace DiamondDetect.Core.Models;

/// <summary>
/// 与 inference_common / UI 共用的检测结果契约。
/// </summary>
public sealed class DetectionResult
{
    public string Path { get; set; } = "";
    public string Class { get; set; } = "";
    public double Confidence { get; set; }
    public string MaxClass { get; set; } = "";
    public double MaxConfidence { get; set; }
    public Dictionary<string, double> AllScores { get; set; } = new();
    public string? TrueClass { get; set; }
    public bool Flagged { get; set; }
    public bool CorrectionSaved { get; set; }
    public bool IsChecked { get; set; } = true;
    public double ElapsedMs { get; set; }
    public bool FromFolderImport { get; set; }
    public string? Error { get; set; }
    public string? Backend { get; set; }
    public string? Device { get; set; }

    public string FileName => System.IO.Path.GetFileName(Path);

    public string ConfidenceText => $"{Confidence * 100:0.0}%";

    public bool IsResultsListItem => !Flagged && !CorrectionSaved;

    public bool IsCorrectionPending => Flagged && !CorrectionSaved;

    public void EnsureMeta()
    {
        Path = NormalizePath(Path);
        TrueClass ??= "";
    }

    public static string NormalizePath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return "";
        try
        {
            return System.IO.Path.GetFullPath(path.Trim());
        }
        catch
        {
            return path.Trim();
        }
    }
}
