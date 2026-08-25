using System.Globalization;
using DiamondDetect.Core.Abstractions;
using DiamondDetect.Core.Models;
using Python.Runtime;

namespace DiamondDetect.Bridge;

/// <summary>
/// 通过 pythonnet 调用 inference_engine / inference_engine_onnx。
/// </summary>
public sealed class PythonInferenceEngine : IInferenceEngine
{
    private readonly IPythonRuntimeHost _host;
    private readonly bool _deployMode;
    private PyObject? _engine;

    public PythonInferenceEngine(IPythonRuntimeHost host, bool deployMode)
    {
        _host = host;
        _deployMode = deployMode;
    }

    public bool IsLoaded { get; private set; }
    public string Backend { get; private set; } = "none";
    public string Device { get; private set; } = "cpu";
    public IReadOnlyList<string> Classes { get; private set; } = Array.Empty<string>();

    /// <summary>底层 Python InferenceEngine，供 SAHI 流水线复用同一实例。</summary>
    public PyObject? NativeEngine => _engine;

    public string Load(string? ptPath, string? onnxPath, bool useGpu)
    {
        _host.Initialize();
        using (Py.GIL())
        {
            var moduleName = _deployMode ? "inference_engine_onnx" : "inference_engine";
            dynamic mod = Py.Import(moduleName);
            _engine?.Dispose();
            _engine = (PyObject)mod.InferenceEngine();
            dynamic eng = _engine;

            PyObject? pt = ToPyPathOrNone(ptPath);
            PyObject? onnx = ToPyPathOrNone(onnxPath);
            string msg = (string)eng.load(pt, onnx, useGpu);

            IsLoaded = (bool)eng.loaded;
            Backend = eng.backend?.ToString() ?? "unknown";
            Device = eng.device?.ToString() ?? "cpu";

            var classes = new List<string>();
            foreach (var c in eng.classes)
                classes.Add(c.ToString());
            Classes = classes;
            return msg;
        }
    }

    public DetectionResult Predict(string imagePath)
    {
        EnsureLoaded();
        using (Py.GIL())
        {
            dynamic eng = _engine!;
            dynamic raw = eng.predict(imagePath);
            return MapResult(raw);
        }
    }

    public IReadOnlyList<DetectionResult> PredictBatch(
        IReadOnlyList<string> imagePaths,
        IProgress<(int current, int total)>? progress = null,
        CancellationToken cancellationToken = default)
    {
        EnsureLoaded();
        progress?.Report((0, imagePaths.Count));

        // Phase 1：先保证结果正确；进度/取消在后续用 PyObject 回调完善。
        // 取消：分批逐张 Predict，便于响应 CancellationToken。
        if (cancellationToken.CanBeCanceled || progress != null)
        {
            var list = new List<DetectionResult>(imagePaths.Count);
            for (var i = 0; i < imagePaths.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                list.Add(Predict(imagePaths[i]));
                progress?.Report((i + 1, imagePaths.Count));
            }
            return list;
        }

        using (Py.GIL())
        {
            dynamic eng = _engine!;
            var pyList = new PyList();
            foreach (var p in imagePaths)
                pyList.Append(new PyString(p));

            dynamic rawList = eng.predict_batch(pyList);
            var results = new List<DetectionResult>();
            foreach (var item in rawList)
                results.Add(MapResult(item));
            progress?.Report((results.Count, results.Count));
            return results;
        }
    }

    public void Dispose()
    {
        _engine?.Dispose();
        _engine = null;
        IsLoaded = false;
    }

    private void EnsureLoaded()
    {
        if (!IsLoaded || _engine is null)
            throw new InvalidOperationException("分类引擎未加载。请先在设置中加载模型或等待启动自动加载。");
    }

    private static PyObject ToPyPathOrNone(string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return Runtime.None!;
        return new PyString(path);
    }

    internal static DetectionResult MapResult(dynamic raw)
    {
        var r = new DetectionResult
        {
            Path = GetStr(raw, "path"),
            Class = GetStr(raw, "class"),
            Confidence = GetDouble(raw, "confidence"),
            MaxClass = GetStr(raw, "max_class"),
            MaxConfidence = GetDouble(raw, "max_confidence"),
            TrueClass = NullIfEmpty(GetStr(raw, "true_class")),
            Flagged = GetBool(raw, "flagged"),
            Error = NullIfEmpty(GetStr(raw, "error")),
            Backend = NullIfEmpty(GetStr(raw, "backend")),
            Device = NullIfEmpty(GetStr(raw, "device")),
            ElapsedMs = GetDouble(raw, "elapsed_ms"),
        };

        try
        {
            if (HasKey(raw, "all_scores"))
            {
                dynamic scores = raw["all_scores"];
                foreach (var key in scores)
                {
                    var k = key.ToString();
                    r.AllScores[k!] = Convert.ToDouble(scores[k], CultureInfo.InvariantCulture);
                }
            }
        }
        catch
        {
            // scores optional
        }

        r.EnsureMeta();
        return r;
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
            if (!HasKey(raw, key))
                return "";
            var v = raw[key];
            if (v is null)
                return "";
            if (v is PyObject py && py.IsNone())
                return "";
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
            if (!HasKey(raw, key) || raw[key] is null)
                return 0;
            return Convert.ToDouble(raw[key], CultureInfo.InvariantCulture);
        }
        catch
        {
            return 0;
        }
    }

    private static bool GetBool(dynamic raw, string key)
    {
        try
        {
            if (!HasKey(raw, key) || raw[key] is null)
                return false;
            return Convert.ToBoolean(raw[key]);
        }
        catch
        {
            return false;
        }
    }

    private static string? NullIfEmpty(string s) => string.IsNullOrWhiteSpace(s) ? null : s;
}
