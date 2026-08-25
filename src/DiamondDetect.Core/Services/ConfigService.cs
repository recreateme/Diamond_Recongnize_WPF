using System.Text.Json;
using DiamondDetect.Core.Configuration;

namespace DiamondDetect.Core.Services;

public interface IConfigService
{
    AppConfig Current { get; }
    string ConfigFilePath { get; }
    AppConfig Load();
    void Save(AppConfig config);
}

/// <summary>
/// 读写根目录 app_config.json，字段与原 PyQt 版兼容。
/// </summary>
public sealed class ConfigService : IConfigService
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };

    private readonly bool _deployMode;
    private readonly string _rootDir;

    public ConfigService(string rootDir, bool deployMode)
    {
        _rootDir = rootDir;
        _deployMode = deployMode;
        ConfigFilePath = Path.Combine(_rootDir, "app_config.json");
        Current = AppConfig.CreateDefault(deployMode);
    }

    public AppConfig Current { get; private set; }

    public string ConfigFilePath { get; }

    public AppConfig Load()
    {
        var defaults = AppConfig.CreateDefault(_deployMode);
        if (!File.Exists(ConfigFilePath))
        {
            Current = Heal(_rootDir, defaults, defaults);
            if (_deployMode)
                TryPersistHealed(Current);
            return Current;
        }

        try
        {
            var json = File.ReadAllText(ConfigFilePath);
            var loaded = JsonSerializer.Deserialize<AppConfig>(json, JsonOptions) ?? defaults;
            Current = Heal(appRoot: _rootDir, defaults, loaded);
            if (_deployMode)
                TryPersistHealed(Current);
            return Current;
        }
        catch (Exception)
        {
            Current = defaults;
            return Current;
        }
    }

    public void Save(AppConfig config)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(ConfigFilePath)!);
        var json = JsonSerializer.Serialize(config, JsonOptions);
        File.WriteAllText(ConfigFilePath, json);
        Current = config;
    }

    private void TryPersistHealed(AppConfig config)
    {
        try { Save(config); }
        catch { /* 只读目录时跳过 */ }
    }

    private static AppConfig Heal(string appRoot, AppConfig defaults, AppConfig loaded)
    {
        loaded.OnnxPath = string.IsNullOrWhiteSpace(loaded.OnnxPath) ? defaults.OnnxPath : loaded.OnnxPath;
        if (string.IsNullOrWhiteSpace(loaded.YoloPath) || !File.Exists(PackLayout.ResolveYolo(appRoot, loaded.YoloPath)))
        {
            var found = PackLayout.ResolveYolo(appRoot, PackLayout.YoloRelative);
            loaded.YoloPath = File.Exists(found) ? PackLayout.YoloRelative : (loaded.YoloPath ?? defaults.YoloPath);
        }
        else if (Path.IsPathRooted(loaded.YoloPath))
        {
            var packaged = PackLayout.YoloWeights(appRoot);
            if (File.Exists(packaged))
                loaded.YoloPath = PackLayout.YoloRelative;
        }

        if (string.IsNullOrWhiteSpace(loaded.OnnxPath) || !File.Exists(PackLayout.ResolveOnnx(appRoot, loaded.OnnxPath)))
        {
            var found = PackLayout.ResolveOnnx(appRoot, PackLayout.OnnxRelative);
            if (File.Exists(found))
                loaded.OnnxPath = PackLayout.OnnxRelative;
        }

        if (string.IsNullOrWhiteSpace(loaded.SahiOutputDir))
            loaded.SahiOutputDir = defaults.SahiOutputDir;
        if (string.IsNullOrWhiteSpace(loaded.SahiDevice))
            loaded.SahiDevice = "auto";
        return loaded;
    }
}
