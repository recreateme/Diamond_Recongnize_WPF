using System.Globalization;
using System.Text;
using DiamondDetect.Core.Abstractions;

namespace DiamondDetect.Core.Services;

/// <summary>产品输出根目录 uniformity_summary.csv。</summary>
public static class UniformitySummaryCsv
{
    public static readonly string[] Columns =
    {
        "图像",
        "n_points",
        "voronoi_area_cv_normalized",
        "nn_distance_cv_normalized",
        "clark_evans_R",
        "delaunay_edge_cv_normalized",
        "grid_density_cv",
        "status",
        "conf_filter",
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
                .Append(Fmt(r.VoronoiAreaCvNormalized)).Append(',')
                .Append(Fmt(r.NnDistanceCvNormalized)).Append(',')
                .Append(Fmt(r.ClarkEvansR)).Append(',')
                .Append(Fmt(r.DelaunayEdgeCvNormalized)).Append(',')
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
