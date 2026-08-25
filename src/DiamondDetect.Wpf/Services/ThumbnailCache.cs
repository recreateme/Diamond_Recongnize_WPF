using System.IO;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace DiamondDetect.Wpf.Services;

/// <summary>
/// 解码限幅的缩略图 LRU 缓存，避免结果表刷新时反复全分辨率解码。
/// </summary>
public sealed class ThumbnailCache
{
    public static ThumbnailCache Shared { get; } = new(capacity: 256);

    private readonly int _capacity;
    private readonly object _gate = new();
    private readonly Dictionary<string, LinkedListNode<CacheEntry>> _map = new(StringComparer.OrdinalIgnoreCase);
    private readonly LinkedList<CacheEntry> _lru = new();

    public ThumbnailCache(int capacity = 256)
    {
        _capacity = Math.Max(32, capacity);
    }

    public BitmapImage? GetOrLoad(string? path, int logicalDecodeWidth)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return null;

        var decodeWidth = ScaleDecodeWidth(logicalDecodeWidth);
        long ticks;
        try { ticks = File.GetLastWriteTimeUtc(path).Ticks; }
        catch { return null; }

        var key = $"{path}|{decodeWidth}|{ticks}";
        lock (_gate)
        {
            if (_map.TryGetValue(key, out var node))
            {
                _lru.Remove(node);
                _lru.AddFirst(node);
                return node.Value.Image;
            }
        }

        var image = Decode(path, decodeWidth);
        if (image is null) return null;

        lock (_gate)
        {
            if (_map.TryGetValue(key, out var existing))
            {
                _lru.Remove(existing);
                _lru.AddFirst(existing);
                return existing.Value.Image;
            }

            var entry = new CacheEntry(key, image);
            var node = _lru.AddFirst(entry);
            _map[key] = node;
            while (_map.Count > _capacity && _lru.Last is not null)
            {
                var last = _lru.Last;
                _lru.RemoveLast();
                _map.Remove(last.Value.Key);
            }
            return image;
        }
    }

    public void Clear()
    {
        lock (_gate)
        {
            _map.Clear();
            _lru.Clear();
        }
    }

    public static int ScaleDecodeWidth(int logicalPixels)
    {
        var scale = 1.0;
        try
        {
            if (Application.Current?.MainWindow is { } win)
                scale = VisualTreeHelper.GetDpi(win).DpiScaleX;
        }
        catch
        {
            // ignore — fall back to 1.0
        }
        return Math.Max(16, (int)Math.Ceiling(logicalPixels * Math.Max(1.0, scale)));
    }

    private static BitmapImage? Decode(string path, int decodeWidth)
    {
        try
        {
            var bi = new BitmapImage();
            bi.BeginInit();
            bi.CacheOption = BitmapCacheOption.OnLoad;
            bi.CreateOptions = BitmapCreateOptions.IgnoreColorProfile;
            bi.UriSource = new Uri(path, UriKind.Absolute);
            bi.DecodePixelWidth = decodeWidth;
            bi.EndInit();
            bi.Freeze();
            return bi;
        }
        catch
        {
            return null;
        }
    }

    private sealed record CacheEntry(string Key, BitmapImage Image);
}
