using System.Collections.ObjectModel;
using System.Windows.Media;
using CommunityToolkit.Mvvm.ComponentModel;

namespace DiamondDetect.Wpf.ViewModels;

public partial class ScoreBarItem : ObservableObject
{
    public string Name { get; init; } = "";
    public double Score { get; init; }
    public string ScoreText => $"{Score * 100:0.0}%";
    public int Percent => (int)Math.Round(Score * 100);
    public bool IsPredicted { get; init; }
    public Brush BarBrush => IsPredicted
        ? new SolidColorBrush(Color.FromRgb(0x19, 0x76, 0xD2))
        : new SolidColorBrush(Color.FromRgb(0xB0, 0xBE, 0xC5));
}

public partial class BatchRowItem : ObservableObject
{
    public string FileName { get; init; } = "";
    public string Class { get; init; } = "";
    public string ConfidenceText { get; init; } = "";
    public string ElapsedText { get; init; } = "";
}
