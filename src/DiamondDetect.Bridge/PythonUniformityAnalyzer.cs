using System.Globalization;
using DiamondDetect.Core.Abstractions;
using DiamondDetect.Core.Services;
using Python.Runtime;

namespace DiamondDetect.Bridge;

/// <summary>
/// 调用 python_core.diamond_uniformity：扫描 detect_boxes/result.json 并落盘分数/可视化。
/// </summary>
public sealed class PythonUniformityAnalyzer : IUniformityAnalyzer
{
    private readonly IPythonRuntimeHost _host;

    public PythonUniformityAnalyzer(IPythonRuntimeHost host)
    {
        _host = host;
    }

    public Task<IReadOnlyList<UniformityTileInfo>> ListTilesAsync(
        string outputRoot,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => ListTiles(outputRoot, cancellationToken), cancellationToken);
    }

    public Task<UniformityBatchResult> AnalyzeOutputRootAsync(
        string outputRoot,
        double confThreshold,
        IReadOnlyList<string>? jsonPaths = null,
        bool writeVisualization = false,
        IProgress<(int current, int total, string message)>? progress = null,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(
            () => AnalyzeOutputRoot(
                outputRoot, confThreshold, jsonPaths, writeVisualization, progress, cancellationToken),
            cancellationToken);
    }

    public Task<UniformityVisResult> RenderVisualizationAsync(
        string jsonPath,
        double confThreshold,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => RenderVisualization(jsonPath, confThreshold, cancellationToken), cancellationToken);
    }

    private IReadOnlyList<UniformityTileInfo> ListTiles(
        string outputRoot,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(outputRoot) || !Directory.Exists(outputRoot))
            throw new DirectoryNotFoundException("输出目录不存在: " + outputRoot);

        _host.Initialize();
        cancellationToken.ThrowIfCancellationRequested();

        using (Py.GIL())
        {
            dynamic mod = Py.Import("diamond_uniformity");
            dynamic rawList = mod.list_tile_jsons(outputRoot);
            var n = (int)rawList.__len__();
            var items = new List<UniformityTileInfo>(n);
            for (var i = 0; i < n; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                dynamic item = rawList[i];
                items.Add(new UniformityTileInfo
                {
                    Image = GetStr(item, "image"),
                    JsonPath = GetStr(item, "json_path"),
                    SourceKind = GetStr(item, "source_kind"),
                });
            }
            return items;
        }
    }

    private UniformityBatchResult AnalyzeOutputRoot(
        string outputRoot,
        double confThreshold,
        IReadOnlyList<string>? jsonPaths,
        bool writeVisualization,
        IProgress<(int current, int total, string message)>? progress,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(outputRoot) || !Directory.Exists(outputRoot))
            throw new DirectoryNotFoundException("输出目录不存在: " + outputRoot);

        _host.Initialize();
        cancellationToken.ThrowIfCancellationRequested();

        List<string> paths;
        if (jsonPaths is { Count: > 0 })
        {
            paths = jsonPaths.Where(p => !string.IsNullOrWhiteSpace(p)).ToList();
        }
        else
        {
            paths = ListTiles(outputRoot, cancellationToken)
                .Select(t => t.JsonPath)
                .Where(p => !string.IsNullOrWhiteSpace(p))
                .ToList();
        }

        var rows = new List<UniformityScoreRow>(paths.Count);
        var total = Math.Max(1, paths.Count);
        progress?.Report((0, total, paths.Count == 0 ? "未找到可分析的 JSON" : "正在计算均匀度…"));

        if (paths.Count > 0)
        {
            using (Py.GIL())
            {
                dynamic mod = Py.Import("diamond_uniformity");
                for (var i = 0; i < paths.Count; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    progress?.Report((i, paths.Count, $"均匀度 {i + 1}/{paths.Count}"));
                    dynamic item = mod.analyze_detect_boxes_file(
                        paths[i],
                        conf_threshold: confThreshold,
                        write_scores: true,
                        write_vis: writeVisualization);
                    rows.Add(MapRow(item));
                }
                progress?.Report((paths.Count, paths.Count, $"均匀度 {paths.Count}/{paths.Count}"));
            }
        }

        string? summaryCsv = null;
        if (rows.Count > 0)
            summaryCsv = UniformitySummaryCsv.Write(outputRoot, rows);

        return new UniformityBatchResult
        {
            OutputRoot = Path.GetFullPath(outputRoot),
            SummaryCsv = summaryCsv,
            Rows = rows,
        };
    }

    private UniformityVisResult RenderVisualization(
        string jsonPath,
        double confThreshold,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(jsonPath) || !File.Exists(jsonPath))
            throw new FileNotFoundException("检测 JSON 不存在: " + jsonPath);

        _host.Initialize();
        cancellationToken.ThrowIfCancellationRequested();

        using (Py.GIL())
        {
            dynamic mod = Py.Import("diamond_uniformity");
            dynamic raw = mod.render_uniformity_visualization(
                jsonPath,
                conf_threshold: confThreshold);
            return new UniformityVisResult
            {
                VisPath = GetStr(raw, "vis_path"),
                Status = GetStr(raw, "status"),
                Background = GetStr(raw, "background"),
                NPoints = (int)GetDouble(raw, "n_points"),
            };
        }
    }

    private static UniformityScoreRow MapRow(dynamic raw)
    {
        var sourceJson = GetStr(raw, "source_json");
        var visPath = GetStr(raw, "vis_path");
        if (string.IsNullOrEmpty(visPath) && !string.IsNullOrEmpty(sourceJson))
        {
            var candidate = Path.Combine(Path.GetDirectoryName(sourceJson) ?? "", "uniformity_vis.jpg");
            if (File.Exists(candidate))
                visPath = candidate;
        }

        return new UniformityScoreRow
        {
            Image = GetStr(raw, "image"),
            SourceJson = sourceJson,
            ScoresJson = GetStr(raw, "scores_json"),
            VisPath = visPath,
            SourceKind = GetStr(raw, "source_kind"),
            NPoints = (int)GetDouble(raw, "n_points"),
            VoronoiAreaCvNormalized = GetNullableDouble(raw, "voronoi_area_cv_normalized"),
            NnDistanceCvNormalized = GetNullableDouble(raw, "nn_distance_cv_normalized"),
            ClarkEvansR = GetNullableDouble(raw, "clark_evans_R"),
            DelaunayEdgeCvNormalized = GetNullableDouble(raw, "delaunay_edge_cv_normalized"),
            GridDensityCv = GetNullableDouble(raw, "grid_density_cv"),
            Status = GetStr(raw, "status"),
            ConfFilter = GetStr(raw, "conf_filter"),
        };
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
        catch
        {
            return "";
        }
    }

    private static double GetDouble(dynamic raw, string key)
    {
        try
        {
            if (!HasKey(raw, key) || raw[key] is null) return 0;
            var v = raw[key];
            if (v is PyObject py && py.IsNone()) return 0;
            return Convert.ToDouble(v, CultureInfo.InvariantCulture);
        }
        catch
        {
            return 0;
        }
    }

    private static double? GetNullableDouble(dynamic raw, string key)
    {
        try
        {
            if (!HasKey(raw, key) || raw[key] is null) return null;
            var v = raw[key];
            if (v is PyObject py && py.IsNone()) return null;
            return Convert.ToDouble(v, CultureInfo.InvariantCulture);
        }
        catch
        {
            return null;
        }
    }
}
