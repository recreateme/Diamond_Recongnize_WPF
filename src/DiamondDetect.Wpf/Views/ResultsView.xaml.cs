using System.Windows.Controls;
using DiamondDetect.Wpf.ViewModels;

namespace DiamondDetect.Wpf.Views;

public partial class ResultsView : UserControl
{
    public ResultsView()
    {
        InitializeComponent();
        IsVisibleChanged += (_, _) =>
        {
            if (IsVisible && DataContext is ResultsViewModel vm)
                vm.Refresh();
        };
    }
}
