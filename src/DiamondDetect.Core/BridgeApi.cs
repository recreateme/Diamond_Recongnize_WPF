namespace DiamondDetect.Core;

/// <summary>
/// C# Bridge 与 python_core 之间的接口版本标记。
/// 变更推理结果字段、配置契约或 Bridge 方法签名时请递增。
/// </summary>
public static class BridgeApi
{
    public const string Version = "1.1.0";
    public const string PythonCoreContract =
        "inference_common.result_dict + app_config.json + diamond_uniformity";
}
