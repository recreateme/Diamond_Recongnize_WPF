using System.Globalization;
using System.Text;
using DiamondDetect.Core.Models;

namespace DiamondDetect.Core.Services;

public sealed class ExportService
{
    public string ExportCsv(IEnumerable<DetectionResult> results, string filePath)
    {
        var list = results.Where(r => r.IsResultsListItem || r.IsCorrectionPending).ToList();
        var sb = new StringBuilder();
        sb.AppendLine("path,file_name,class,confidence,max_class,max_confidence,flagged,true_class,elapsed_ms");
        foreach (var r in list)
        {
            sb.Append(Csv(r.Path)).Append(',')
              .Append(Csv(r.FileName)).Append(',')
              .Append(Csv(r.Class)).Append(',')
              .Append(r.Confidence.ToString("0.######", CultureInfo.InvariantCulture)).Append(',')
              .Append(Csv(r.MaxClass)).Append(',')
              .Append(r.MaxConfidence.ToString("0.######", CultureInfo.InvariantCulture)).Append(',')
              .Append(r.Flagged ? "1" : "0").Append(',')
              .Append(Csv(r.TrueClass ?? "")).Append(',')
              .Append(r.ElapsedMs.ToString("0.###", CultureInfo.InvariantCulture))
              .AppendLine();
        }
        var dir = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);
        File.WriteAllText(filePath, sb.ToString(), new UTF8Encoding(encoderShouldEmitUTF8Identifier: true));
        return filePath;
    }

    /// <summary>按预测类别分子文件夹复制图像；返回导出文件数。</summary>
    public int ExportClassifiedImages(
        IEnumerable<DetectionResult> results,
        string exportRoot,
        bool onlyChecked)
    {
        Directory.CreateDirectory(exportRoot);
        var count = 0;
        foreach (var r in results)
        {
            if (!r.IsResultsListItem) continue;
            if (onlyChecked && !r.IsChecked) continue;
            if (string.IsNullOrWhiteSpace(r.Class) || r.Class == "ERROR") continue;
            if (!File.Exists(r.Path)) continue;

            var classDir = Path.Combine(exportRoot, SanitizeFolderName(r.Class));
            Directory.CreateDirectory(classDir);
            var dest = UniqueDest(classDir, Path.GetFileName(r.Path));
            File.Copy(r.Path, dest, overwrite: false);
            count++;
        }
        return count;
    }

    public int ArchiveToCorrections(
        ResultStore store,
        IReadOnlyList<int> indices,
        string correctionsRoot,
        string trueClass)
    {
        var existing = indices
            .Distinct()
            .Where(i =>
            {
                var r = store.GetAt(i);
                return r != null && File.Exists(r.Path);
            })
            .ToList();

        return store.ArchiveAndRemove(existing, r =>
        {
            var dstDir = Path.Combine(correctionsRoot, SanitizeFolderName(trueClass));
            Directory.CreateDirectory(dstDir);
            var dest = UniqueDest(dstDir, Path.GetFileName(r.Path));
            File.Copy(r.Path, dest, overwrite: false);
            r.TrueClass = trueClass;
            r.CorrectionSaved = true;
        });
    }

    public static string SanitizeFolderName(string name)
    {
        var s = (name ?? "").Trim();
        foreach (var ch in Path.GetInvalidFileNameChars())
            s = s.Replace(ch, '_');
        return string.IsNullOrWhiteSpace(s) ? "unknown" : s;
    }

    private static string UniqueDest(string dir, string filename)
    {
        var dest = Path.Combine(dir, filename);
        if (!File.Exists(dest)) return dest;
        var stem = Path.GetFileNameWithoutExtension(filename);
        var ext = Path.GetExtension(filename);
        var n = 1;
        while (File.Exists(dest))
        {
            dest = Path.Combine(dir, $"{stem}_{n}{ext}");
            n++;
        }
        return dest;
    }

    private static string Csv(string value)
    {
        if (value.Contains('"') || value.Contains(',') || value.Contains('\n'))
            return "\"" + value.Replace("\"", "\"\"") + "\"";
        return value;
    }
}
