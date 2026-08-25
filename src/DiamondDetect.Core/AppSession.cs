using DiamondDetect.Core.Abstractions;
using DiamondDetect.Core.Configuration;
using DiamondDetect.Core.Services;

namespace DiamondDetect.Core;

/// <summary>共享会话，对齐原 AppState。</summary>
public sealed class AppSession
{
    public const string AdminPagePassword = "20250508";
    public const string AppVersion = "v0.1-wpf";

    public AppSession(
        DeployMode deployMode,
        IConfigService configService,
        ResultStore results,
        string appRoot)
    {
        DeployMode = deployMode;
        ConfigService = configService;
        Results = results;
        AppRoot = appRoot;
        Config = configService.Load();
    }

    public DeployMode DeployMode { get; }
    public bool IsDeploy => DeployMode.IsDeploy();
    public string AppRoot { get; }
    public IConfigService ConfigService { get; }
    public AppConfig Config { get; set; }
    public ResultStore Results { get; }
    public IInferenceEngine? Engine { get; set; }
    public bool AdminUnlocked { get; set; }
    public string StatusText { get; set; } = "就绪";
    public string TitleSuffix => IsDeploy ? " · 机台版" : "";
}
