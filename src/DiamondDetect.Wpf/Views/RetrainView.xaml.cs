using System.Windows.Controls;
using DiamondDetect.Wpf.ViewModels;

namespace DiamondDetect.Wpf.Views;

public partial class RetrainView : UserControl
{
    public RetrainView()
    {
        InitializeComponent();
        IsVisibleChanged += (_, _) =>
        {
            if (IsVisible && DataContext is RetrainViewModel vm)
                vm.RefreshStatsCommand.Execute(null);
        };
    }
}
