using System.Windows;
using DiamondDetect.Wpf.ViewModels;

namespace DiamondDetect.Wpf;

public partial class MainWindow : Window
{
    public MainWindow(
        MainViewModel vm,
        DiamondDetectViewModel diamondVm,
        DetectionViewModel detectionVm,
        ResultsViewModel resultsVm,
        CorrectionViewModel correctionVm,
        RetrainViewModel retrainVm)
    {
        InitializeComponent();
        DataContext = vm;
        DiamondViewHost.DataContext = diamondVm;
        DetectViewHost.DataContext = detectionVm;
        ResultsViewHost.DataContext = resultsVm;
        CorrectionViewHost.DataContext = correctionVm;
        RetrainViewHost.DataContext = retrainVm;

        vm.AttachPageActions(
            refreshResults: () => resultsVm.Refresh(),
            refreshCorrection: () => correctionVm.Refresh(),
            stopDetection: () => detectionVm.StopBatchCommand.Execute(null),
            stopDiamond: () => diamondVm.StopCommand.Execute(null),
            stopTrain: () => retrainVm.StopTrainCommand.Execute(null));

        Loaded += async (_, _) => await vm.InitializeAsync();
    }
}
