namespace DiamondDetect.Bridge;

/// <summary>供 Python.NET 将 CancellationToken 暴露为 should_stop() 回调。</summary>
public sealed class CancellationBridge
{
    private readonly CancellationToken _token;

    public CancellationBridge(CancellationToken token) => _token = token;

    public bool ShouldStop() => _token.IsCancellationRequested;
}
