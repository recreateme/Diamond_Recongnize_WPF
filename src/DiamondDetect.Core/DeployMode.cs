namespace DiamondDetect.Core;

/// <summary>开发版完整功能 / 机台 ONNX 模式（对齐 DEFECTS_DEPLOY）。</summary>
public enum DeployMode
{
    Full = 0,
    OnnxDeploy = 1,
}

public static class DeployModeDetector
{
    public static DeployMode Detect(string? appRoot = null)
    {
        var env = Environment.GetEnvironmentVariable("DEFECTS_DEPLOY");
        if (string.Equals(env, "1", StringComparison.Ordinal))
            return DeployMode.OnnxDeploy;
        if (!string.IsNullOrWhiteSpace(appRoot) && PackLayout.LooksLikeMachinePack(appRoot))
            return DeployMode.OnnxDeploy;
        return DeployMode.Full;
    }

    public static bool IsDeploy(this DeployMode mode) => mode == DeployMode.OnnxDeploy;
}
