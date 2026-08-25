using System.Windows.Controls;
using DiamondDetect.Wpf.ViewModels;

namespace DiamondDetect.Wpf.Views;

public partial class CorrectionView : UserControl
{
    public CorrectionView()
    {
        InitializeComponent();
        IsVisibleChanged += (_, _) =>
        {
            if (IsVisible && DataContext is CorrectionViewModel vm)
                vm.Refresh();
        };
    }
}
