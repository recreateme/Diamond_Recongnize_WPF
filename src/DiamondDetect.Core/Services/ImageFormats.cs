using DiamondDetect.Core.Models;

namespace DiamondDetect.Core.Services;

public static class ImageFormats
{
    public static readonly HashSet<string> Extensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
    };

    public static bool IsImage(string path) =>
        Extensions.Contains(System.IO.Path.GetExtension(path));

    public static IReadOnlyList<string> EnumerateImages(string folder, bool recursive = true)
    {
        if (!Directory.Exists(folder))
            return Array.Empty<string>();

        var option = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
        return Directory.EnumerateFiles(folder, "*.*", option)
            .Where(IsImage)
            .Select(DetectionResult.NormalizePath)
            .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }
}
