using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using DiamondDetect.Core.Services;
using DiamondDetect.Wpf.ViewModels;

namespace DiamondDetect.Wpf.Views;

public partial class DetectionView : UserControl
{
    public DetectionView()
    {
        InitializeComponent();
    }

    private DetectionViewModel? Vm => DataContext as DetectionViewModel;

    private void DropZone_Click(object sender, MouseButtonEventArgs e)
        => Vm?.BrowseSingleImageCommand.Execute(null);

    private void OnDragOver(object sender, DragEventArgs e)
    {
        e.Effects = e.Data.GetDataPresent(DataFormats.FileDrop) ? DragDropEffects.Copy : DragDropEffects.None;
        e.Handled = true;
    }

    private void OnDrop(object sender, DragEventArgs e)
    {
        if (Vm is null) return;
        if (e.Data.GetData(DataFormats.FileDrop) is not string[] files || files.Length == 0)
            return;
        var path = files[0];
        if (ImageFormats.IsImage(path))
            Vm.SetSingleImage(path);
    }
}
