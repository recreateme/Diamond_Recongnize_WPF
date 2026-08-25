using System.Globalization;
using System.Text;
using DiamondDetect.Core.Abstractions;
using Python.Runtime;

namespace DiamondDetect.Bridge;

/// <summary>
/// 调用 python_core.sahi_detector.SahiDetector / SahiPipeline，对齐原 SahiPipelineWorker。
/// </summary>
public sealed class PythonSahiPipeline : ISahiPipeline
{
    private readonly IPythonRuntimeHost _host;
    private readonly PythonInferenceEngine _classifier;

    public PythonSahiPipeline(IPythonRuntimeHost host, IInferenceEngine classifier)
    {
        _host = host;
        _classifier = classifier as PythonInferenceEngine
            ?? throw new ArgumentException("SAHI 需要 PythonInferenceEngine 作为分类器。", nameof(classifier));
    }

    public Task<IReadOnlyList<SahiImageStats>> ProcessImagesAsync(
        IReadOnlyList<string> imagePaths,
        SahiRunOptions options,
        IProgress<SahiProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => ProcessImages(imagePaths, options, progress, cancellationToken), cancellationToken);
    }

    private IReadOnlyList<SahiImageStats> ProcessImages(
        IReadOnlyList<string> imagePaths,
        SahiRunOptions options,
        IProgress<SahiProgress>? progress,
        CancellationToken cancellationToken)
    {
        if (!_classifier.IsLoaded || _classifier.NativeEngine is null)
            throw new InvalidOperationException("分类引擎未加载，请先在「设置 → 分类配置」中加载分类模型。");
        if (string.IsNullOrWhiteSpace(options.YoloPath) || !File.Exists(options.YoloPath))
            throw new FileNotFoundException("未配置有效的 YOLO 模型。", options.YoloPath);
        if (imagePaths.Count == 0)
            throw new ArgumentException("未选择输入图像。");

        Directory.CreateDirectory(options.OutputDir);
        _host.Initialize();

        var allStats = new List<SahiImageStats>();
        var total = imagePaths.Count;

        using (Py.GIL())
        {
            dynamic sahi = Py.Import("sahi_detector");
            dynamic detector = sahi.SahiDetector(
                options.YoloPath,
                options.Device,
                options.SliceSize,
                options.Overlap,
                options.DetConf,
                options.BatchSize,
                0.50,
                options.IosThresh,
                options.MinAreaRatio,
                null,
                options.MaxAspectRatio,
                options.EdgeFilter,
                options.EdgeMarginPx);

            string loadMsg = (string)detector.load();
            progress?.Report(new SahiProgress
            {
                Current = 0,
                Total = total,
                Message = loadMsg,
            });

            dynamic pipeline = sahi.SahiPipeline(
                detector,
                _classifier.NativeEngine,
                options.OutputDir,
                options.CropPadding);

            for (var i = 0; i < imagePaths.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var imgPath = imagePaths[i];
                var name = Path.GetFileName(imgPath);
                progress?.Report(new SahiProgress
                {
                    Current = i,
                    Total = total,
                    Message = $"[{i + 1}/{total}] {name}",
                });

                SahiImageStats stats;
                try
                {
                    dynamic raw = pipeline.process_image(imgPath);
                    stats = MapStats(raw);
                }
                catch (Exception ex)
                {
                    stats = new SahiImageStats
                    {
                        Image = name,
                        Error = ex.Message,
                        TotalDiamonds = 0,
                    };
                }

                allStats.Add(stats);
                progress?.Report(new SahiProgress
                {
                    Current = i + 1,
                    Total = total,
                    Message = string.IsNullOrEmpty(stats.Error)
                        ? $"[{i + 1}/{total}] 完成 {stats.Image} · {stats.TotalDiamonds} 颗"
                        : $"[{i + 1}/{total}] 失败 {stats.Image}",
                    LastImage = stats,
                });
            }
        }

        WriteSummaryCsv(options.OutputDir, allStats);
        progress?.Report(new SahiProgress
        {
            Current = total,
            Total = total,
            Message = cancellationToken.IsCancellationRequested ? "已停止" : "处理完成",
        });
        return allStats;
    }

    private static SahiImageStats MapStats(dynamic raw)
    {
        var s = new SahiImageStats
        {
            Image = GetStr(raw, "image"),
            TotalDiamonds = (int)GetDouble(raw, "total_diamonds"),
            DetectionTimeS = GetDouble(raw, "detection_time_s"),
            ClassificationTimeS = GetDouble(raw, "classification_time_s"),
            TotalTimeS = GetDouble(raw, "total_time_s"),
            OutputDir = GetStr(raw, "output_dir"),
            Error = NullIfEmpty(GetStr(raw, "error")),
            ContainedSkipped = (int)GetDouble(raw, "contained_skipped"),
            SmallSkipped = (int)GetDouble(raw, "small_skipped"),
            AspectSkipped = (int)GetDouble(raw, "aspect_skipped"),
            EdgeSkipped = (int)GetDouble(raw, "edge_skipped"),
        };

        try
        {
            if (HasKey(raw, "defect_counts"))
            {
                dynamic counts = raw["defect_counts"];
                foreach (var key in counts)
                {
                    var k = key.ToString()!;
                    s.DefectCounts[k] = Convert.ToInt32(counts[k], CultureInfo.InvariantCulture);
                }
            }
        }
        catch
        {
            // optional
        }

        return s;
    }

    /// <summary>与应用有效类别一致的固定列顺序（summary.csv 表头稳定）。</summary>
    private static readonly string[] SummaryClassColumns = { "棱边朝上", "点朝上", "面朝上" };

    private static void WriteSummaryCsv(string outputDir, IReadOnlyList<SahiImageStats> allStats)
    {
        if (allStats.Count == 0) return;
        try
        {
            var csvPath = Path.Combine(outputDir, "summary.csv");
            var sb = new StringBuilder();
            sb.Append("图像,汇总钻石数,");
            sb.Append(string.Join(",", SummaryClassColumns));
            sb.AppendLine(",检测耗时(s),分类耗时(s),总耗时(s)");

            var classTotals = new int[SummaryClassColumns.Length];
            var diamondTotal = 0;

            foreach (var s in allStats)
            {
                diamondTotal += s.TotalDiamonds;
                sb.Append(Csv(s.Image)).Append(',').Append(s.TotalDiamonds);
                for (var i = 0; i < SummaryClassColumns.Length; i++)
                {
                    var n = s.DefectCounts.TryGetValue(SummaryClassColumns[i], out var c) ? c : 0;
                    classTotals[i] += n;
                    sb.Append(',').Append(n);
                }
                sb.Append(',')
                  .Append(s.DetectionTimeS.ToString("0.###", CultureInfo.InvariantCulture)).Append(',')
                  .Append(s.ClassificationTimeS.ToString("0.###", CultureInfo.InvariantCulture)).Append(',')
                  .Append(s.TotalTimeS.ToString("0.###", CultureInfo.InvariantCulture))
                  .AppendLine();
            }

            // 多张图才追加一行批次合计；单张不写汇总行
            if (allStats.Count > 1)
            {
                sb.Append(Csv("批次合计")).Append(',').Append(diamondTotal);
                for (var i = 0; i < SummaryClassColumns.Length; i++)
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

    private static string Csv(string value)
    {
        if (value.Contains('"') || value.Contains(',') || value.Contains('\n'))
            return "\"" + value.Replace("\"", "\"\"") + "\"";
        return value;
    }

    private static bool HasKey(dynamic raw, string key)
    {
        try
        {
            var po = (PyObject)raw;
            using var pyKey = new PyString(key);
            return po.InvokeMethod("__contains__", pyKey).IsTrue();
        }
        catch
        {
            return false;
        }
    }

    private static string GetStr(dynamic raw, string key)
    {
        try
        {
            if (!HasKey(raw, key) || raw[key] is null) return "";
            var v = raw[key];
            if (v is PyObject py && py.IsNone()) return "";
            return v.ToString() ?? "";
        }
        catch { return ""; }
    }

    private static double GetDouble(dynamic raw, string key)
    {
        try
        {
            if (!HasKey(raw, key) || raw[key] is null) return 0;
            return Convert.ToDouble(raw[key], CultureInfo.InvariantCulture);
        }
        catch { return 0; }
    }

    private static string? NullIfEmpty(string s) => string.IsNullOrWhiteSpace(s) ? null : s;
}
