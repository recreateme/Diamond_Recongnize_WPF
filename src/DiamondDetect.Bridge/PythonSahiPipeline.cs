using System.Globalization;
using System.Text;
using DiamondDetect.Core.Abstractions;
using DiamondDetect.Core.Services;
using Python.Runtime;

namespace DiamondDetect.Bridge;

/// <summary>
/// 调用 python_core.sahi_detector：完整流水线或 detect-only 定位。
/// </summary>
public sealed class PythonSahiPipeline : ISahiPipeline
{
    private readonly IPythonRuntimeHost _host;
    private readonly PythonInferenceEngine? _classifier;

    public PythonSahiPipeline(IPythonRuntimeHost host, IInferenceEngine classifier)
    {
        _host = host;
        _classifier = classifier as PythonInferenceEngine;
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
        if (string.IsNullOrWhiteSpace(options.YoloPath) || !File.Exists(options.YoloPath))
            throw new FileNotFoundException("未配置有效的 YOLO 模型。", options.YoloPath);
        if (imagePaths.Count == 0)
            throw new ArgumentException("未选择输入图像。");

        if (!options.DetectOnly)
        {
            if (_classifier is null || !_classifier.IsLoaded || _classifier.NativeEngine is null)
                throw new InvalidOperationException("分类引擎未加载，请先在「设置 → 分类配置」中加载分类模型。");
        }

        Directory.CreateDirectory(options.OutputDir);
        _host.Initialize();

        var allStats = new List<SahiImageStats>();
        var total = imagePaths.Count;

        try
        {
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

            dynamic pipeline;
            if (options.DetectOnly)
            {
                pipeline = sahi.SahiDetectOnlyPipeline(
                    detector,
                    options.OutputDir,
                    downsample_enabled: options.DownsampleEnabled,
                    downsample_max_side: options.DownsampleMaxSide,
                    downsample_interpolation: options.DownsampleInterpolation ?? "area",
                    save_visualization: options.SaveVisualization);
            }
            else
            {
                pipeline = sahi.SahiPipeline(
                    detector,
                    _classifier!.NativeEngine,
                    options.OutputDir,
                    options.CropPadding,
                    save_visualization: options.SaveVisualization);
            }

            var cancelBridge = new CancellationBridge(cancellationToken);
            Func<bool> shouldStop = cancelBridge.ShouldStop;

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
                    dynamic raw = pipeline.process_image(imgPath, should_stop: shouldStop);
                    stats = MapStats(raw, options.DetectOnly);
                }
                catch (PythonException ex) when (IsPipelineCancelled(ex))
                {
                    throw new OperationCanceledException(cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    throw;
                }
                catch (Exception ex)
                {
                    stats = new SahiImageStats
                    {
                        Image = name,
                        Error = ex.Message,
                        TotalDiamonds = 0,
                        DetectOnly = options.DetectOnly,
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

        cancellationToken.ThrowIfCancellationRequested();

        if (!options.DetectOnly)
            SahiSummaryCsv.Write(options.OutputDir, allStats);

        progress?.Report(new SahiProgress
        {
            Current = total,
            Total = total,
            Message = cancellationToken.IsCancellationRequested ? "已停止" : "处理完成",
        });
        return allStats;
        }
        catch (OperationCanceledException)
        {
            if (!options.DetectOnly && allStats.Count > 0)
                SahiSummaryCsv.Write(options.OutputDir, allStats);
            throw;
        }
    }

    private static SahiImageStats MapStats(dynamic raw, bool detectOnly)
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
            DetectOnly = detectOnly,
            BoxesJsonPath = NullIfEmpty(GetStr(raw, "boxes_json")),
            BoxesCsvPath = NullIfEmpty(GetStr(raw, "boxes_csv")),
        };

        if (detectOnly)
            return s;

        try
        {
            if (HasKey(raw, "defect_counts") && raw["defect_counts"] is not null)
            {
                dynamic counts = raw["defect_counts"];
                foreach (var key in counts)
                {
                    var k = key.ToString();
                    if (string.IsNullOrEmpty(k)) continue;
                    s.DefectCounts[k] = Convert.ToInt32(counts[key], CultureInfo.InvariantCulture);
                }
            }
        }
        catch
        {
            // optional
        }

        return s;
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

    private static bool IsPipelineCancelled(PythonException ex)
    {
        var typeName = ex.Type?.ToString() ?? "";
        if (typeName.Contains("PipelineCancelledError", StringComparison.Ordinal))
            return true;
        return ex.Message.Contains("PipelineCancelledError", StringComparison.Ordinal);
    }
}
