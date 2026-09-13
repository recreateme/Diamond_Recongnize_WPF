using System.Globalization;
using System.Text;
using DiamondDetect.Core.Abstractions;

namespace DiamondDetect.Core.Services;

/// <summary>
/// 产品输出根目录 uniformity_summary.csv。
/// 表头统一用中文（跟项目其它 CSV 风格一致），CU/DUlq 友好百分比与原始 CV/R
/// 并列——前者给人看，后者留给后续人工标注 + 自动定阈值用，互不替代。
/// 列的定义要和 python_core/diamond_uniformity.py 里的 SUMMARY_COLUMNS 保持
/// 一致（两边各自维护一份，改一边记得改另一边）。
/// </summary>
public static class UniformitySummaryCsv
{
    public static readonly string[] Columns =
    {
        "图像",
        "钻石数",
        "Voronoi均匀度%",
        "Voronoi面积CV",
        "间距均匀度%",
        "规则度R",
        "三角网均匀度%",
        "Delaunay边长CV",
        "区域均匀度%",
        "最差区域均匀度%",
        "区域密度CV",
        "状态",
        "置信度过滤",
    };

    public static string Write(string outputRoot, IReadOnlyList<UniformityScoreRow> rows)
    {
        Directory.CreateDirectory(outputRoot);
        var csvPath = Path.Combine(outputRoot, "uniformity_summary.csv");
        var sb = new StringBuilder();
        sb.AppendLine(string.Join(",", Columns));
        foreach (var r in rows)
        {
            sb.Append(Escape(r.Image)).Append(',')
                .Append(r.NPoints.ToString(CultureInfo.InvariantCulture)).Append(',')
                .Append(Fmt(r.VoronoiAreaCu)).Append(',')
                .Append(Fmt(r.VoronoiAreaCvNormalized)).Append(',')
                .Append(Fmt(r.NnDistanceCu)).Append(',')
                .Append(Fmt(r.ClarkEvansR)).Append(',')
                .Append(Fmt(r.DelaunayEdgeCu)).Append(',')
                .Append(Fmt(r.DelaunayEdgeCvNormalized)).Append(',')
                .Append(Fmt(r.GridDensityCu)).Append(',')
                .Append(Fmt(r.GridDensityDuLq)).Append(',')
                .Append(Fmt(r.GridDensityCv)).Append(',')
                .Append(Escape(r.Status)).Append(',')
                .Append(Escape(r.ConfFilter))
                .AppendLine();
        }
        File.WriteAllText(csvPath, sb.ToString(), new UTF8Encoding(encoderShouldEmitUTF8Identifier: true));
        return csvPath;
    }

    private static string Fmt(double? v) =>
        v is null ? "" : v.Value.ToString("0.######", CultureInfo.InvariantCulture);

    private static string Escape(string value)
    {
        if (string.IsNullOrEmpty(value)) return "";
        if (value.Contains('"') || value.Contains(',') || value.Contains('\n'))
            return "\"" + value.Replace("\"", "\"\"") + "\"";
        return value;
    }
}
