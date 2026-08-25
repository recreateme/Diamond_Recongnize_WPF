using DiamondDetect.Core.Models;

namespace DiamondDetect.Core.Services;

/// <summary>结果列表：同路径 upsert，对齐原 _upsert_result。</summary>
public sealed class ResultStore
{
    private readonly List<DetectionResult> _items = new();
    private readonly object _gate = new();

    public event EventHandler? Changed;

    public IReadOnlyList<DetectionResult> Items
    {
        get
        {
            lock (_gate)
                return _items.ToList();
        }
    }

    public int Count
    {
        get
        {
            lock (_gate)
                return _items.Count;
        }
    }

    public int ResultsListCount
    {
        get
        {
            lock (_gate)
                return _items.Count(r => r.IsResultsListItem);
        }
    }

    public int FlaggedPendingCount
    {
        get
        {
            lock (_gate)
                return _items.Count(r => r.IsCorrectionPending);
        }
    }

    public void Upsert(DetectionResult result)
    {
        result.EnsureMeta();
        lock (_gate)
        {
            var idx = _items.FindIndex(r =>
                string.Equals(r.Path, result.Path, StringComparison.OrdinalIgnoreCase));
            if (idx >= 0)
            {
                var old = _items[idx];
                result.TrueClass = string.IsNullOrEmpty(old.TrueClass) ? result.TrueClass : old.TrueClass;
                result.Flagged = old.Flagged;
                result.CorrectionSaved = old.CorrectionSaved;
                result.IsChecked = old.IsChecked;
                _items[idx] = result;
            }
            else
            {
                _items.Add(result);
            }
        }
        Changed?.Invoke(this, EventArgs.Empty);
    }

    public void Clear()
    {
        lock (_gate)
            _items.Clear();
        Changed?.Invoke(this, EventArgs.Empty);
    }

    public void FlagAt(int index)
    {
        lock (_gate)
        {
            if (index < 0 || index >= _items.Count) return;
            _items[index].Flagged = true;
        }
        Changed?.Invoke(this, EventArgs.Empty);
    }

    public void FlagMany(IEnumerable<int> indices)
    {
        lock (_gate)
        {
            foreach (var index in indices.Distinct())
            {
                if (index < 0 || index >= _items.Count) continue;
                if (_items[index].IsResultsListItem)
                    _items[index].Flagged = true;
            }
        }
        Changed?.Invoke(this, EventArgs.Empty);
    }

    public DetectionResult? GetAt(int index)
    {
        lock (_gate)
            return index >= 0 && index < _items.Count ? _items[index] : null;
    }

    public int IndexOf(DetectionResult item)
    {
        lock (_gate)
            return _items.IndexOf(item);
    }

    public int ArchiveAndRemove(IReadOnlyList<int> indices, Action<DetectionResult> mutateBeforeRemove)
    {
        var saved = 0;
        lock (_gate)
        {
            foreach (var idx in indices.Distinct().OrderByDescending(i => i))
            {
                if (idx < 0 || idx >= _items.Count) continue;
                var r = _items[idx];
                mutateBeforeRemove(r);
                _items.RemoveAt(idx);
                saved++;
            }
        }
        if (saved > 0)
            Changed?.Invoke(this, EventArgs.Empty);
        return saved;
    }

    public void UpsertForceFlag(DetectionResult result)
    {
        result.EnsureMeta();
        result.Flagged = true;
        result.CorrectionSaved = false;
        lock (_gate)
        {
            var idx = _items.FindIndex(r =>
                string.Equals(r.Path, result.Path, StringComparison.OrdinalIgnoreCase));
            if (idx >= 0)
            {
                var old = _items[idx];
                result.IsChecked = old.IsChecked;
                if (string.IsNullOrEmpty(result.TrueClass))
                    result.TrueClass = old.TrueClass;
                _items[idx] = result;
            }
            else
            {
                _items.Add(result);
            }
        }
        Changed?.Invoke(this, EventArgs.Empty);
    }

    public IReadOnlyList<(int Index, DetectionResult Result)> GetPendingCorrections()
    {
        lock (_gate)
        {
            var list = new List<(int, DetectionResult)>();
            for (var i = 0; i < _items.Count; i++)
            {
                if (_items[i].IsCorrectionPending)
                    list.Add((i, _items[i]));
            }
            return list;
        }
    }

    public void SetTrueClass(int index, string trueClass)
    {
        lock (_gate)
        {
            if (index < 0 || index >= _items.Count) return;
            _items[index].TrueClass = trueClass;
        }
    }

    public void NotifyChanged() => Changed?.Invoke(this, EventArgs.Empty);
}
