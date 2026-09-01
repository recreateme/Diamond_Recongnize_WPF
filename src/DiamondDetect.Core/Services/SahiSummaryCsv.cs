using System.Globalization;
using System.Text;
using DiamondDetect.Core.Abstractions;

namespace DiamondDetect.Core.Services;

/// <summary>
/// 钻石检测分类批量输出根目录 summary.csv（与 Bridge 表头一致）。
/// </summary>
public static class SahiSummaryCsv
{
  public static readonly string[] ClassColumns = { "棱边朝上", "点朝上", "面朝上" };

  public static void Write(string outputDir, IReadOnlyList<SahiImageStats> allStats)
  {
    if (string.IsNullOrWhiteSpace(outputDir) || allStats.Count == 0)
      return;

    try
    {
      Directory.CreateDirectory(outputDir);
      var csvPath = Path.Combine(outputDir, "summary.csv");
      var sb = new StringBuilder();
      sb.Append("图像,汇总钻石数,");
      sb.Append(string.Join(",", ClassColumns));
      sb.AppendLine(",检测耗时(s),分类耗时(s),总耗时(s)");

      var classTotals = new int[ClassColumns.Length];
      var diamondTotal = 0;

      foreach (var s in allStats)
      {
        diamondTotal += s.TotalDiamonds;
        sb.Append(Escape(s.Image)).Append(',').Append(s.TotalDiamonds);
        for (var i = 0; i < ClassColumns.Length; i++)
        {
          var n = s.DefectCounts.TryGetValue(ClassColumns[i], out var c) ? c : 0;
          classTotals[i] += n;
          sb.Append(',').Append(n);
        }
        sb.Append(',')
          .Append(s.DetectionTimeS.ToString("0.###", CultureInfo.InvariantCulture)).Append(',')
          .Append(s.ClassificationTimeS.ToString("0.###", CultureInfo.InvariantCulture)).Append(',')
          .Append(s.TotalTimeS.ToString("0.###", CultureInfo.InvariantCulture))
          .AppendLine();
      }

      if (allStats.Count > 1)
      {
        sb.Append(Escape("批次合计")).Append(',').Append(diamondTotal);
        for (var i = 0; i < ClassColumns.Length; i++)
          sb.Append(',').Append(classTotals[i]);
        sb.AppendLine(",,,");
      }

      File.WriteAllText(csvPath, sb.ToString(), new UTF8Encoding(encoderShouldEmitUTF8Identifier: true));
    }
    catch
    {
      // non-fatal
    }
  }

  private static string Escape(string value)
  {
    if (value.Contains('"') || value.Contains(',') || value.Contains('\n'))
      return "\"" + value.Replace("\"", "\"\"") + "\"";
    return value;
  }
}
