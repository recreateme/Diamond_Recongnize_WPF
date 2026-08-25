using System.Windows;
using System.Windows.Controls;
using DiamondDetect.Core;
using DiamondDetect.Wpf.Services;
using DiamondDetect.Wpf.ViewModels;

namespace DiamondDetect.Wpf.Views;

public partial class SettingsPage : UserControl
{
    public SettingsPage()
    {
        InitializeComponent();
        Loaded += (_, _) => RefreshFromSession();
    }

    private MainViewModel? Vm =>
        Window.GetWindow(this)?.DataContext as MainViewModel;

    private void RefreshFromSession()
    {
        var session = Vm?.Session;
        if (session is null) return;

        var c = session.Config;
        PtPathBox.Text = c.PtPath;
        OnnxPathBox.Text = c.OnnxPath;
        UseGpuBox.IsChecked = c.UseGpu;
        LocalDiagBox.IsChecked = c.EnableLocalDiagnostics;
        YoloPathBox.Text = c.YoloPath;
        SahiDeviceBox.Text = c.SahiDevice;
        SliceSizeBox.Text = c.SahiSliceSize.ToString();
        OverlapBox.Text = c.SahiOverlap.ToString("0.##");
        DetConfBox.Text = c.SahiDetConf.ToString("0.##");
        BatchSizeBox.Text = c.SahiBatchSize.ToString();
        PaddingBox.Text = c.SahiCropPadding.ToString();
        OutputDirBox.Text = c.SahiOutputDir;
        IosBox.Text = c.SahiIosThresh.ToString("0.##");
        MinAreaBox.Text = c.SahiMinAreaRatio.ToString("0.##");
        AspectBox.Text = c.SahiMaxAspectRatio.ToString("0.##");
        EdgeFilterBox.IsChecked = c.SahiEdgeFilter;
        EdgeMarginBox.Text = c.SahiEdgeMarginPx.ToString();

        ModeText.Text = session.IsDeploy ? "当前：机台版（ONNX）" : "当前：开发版";
        PtPathBox.IsEnabled = !session.IsDeploy;

        // 机台：默认锁定；开发版：默认可编辑
        if (session.IsDeploy)
            ApplyLockState(session.AdminUnlocked);
        else
            ApplyLockState(true);

        UpdateLockUi(session);
    }

    private void ApplyLockState(bool unlocked)
    {
        FormPanel.IsEnabled = unlocked;
    }

    private void UpdateLockUi(AppSession session)
    {
        if (session.AdminUnlocked)
        {
            LockStatus.Text = "已解锁（管理员）";
            UnlockBtn.Content = "重新锁定";
        }
        else
        {
            LockStatus.Text = session.IsDeploy ? "已锁定 — 输入管理员密码后可修改" : "可编辑（开发版默认开放）";
            UnlockBtn.Content = "输入密码解锁";
        }
    }

    private void Unlock_Click(object sender, RoutedEventArgs e)
    {
        var session = Vm?.Session;
        if (session is null) return;

        if (session.AdminUnlocked)
        {
            session.AdminUnlocked = false;
            ApplyLockState(false);
            UpdateLockUi(session);
            return;
        }

        var dlg = new PasswordPromptWindow { Owner = Window.GetWindow(this) };
        if (dlg.ShowDialog() == true)
        {
            if (dlg.Password == AppSession.AdminPagePassword)
            {
                session.AdminUnlocked = true;
                ApplyLockState(true);
                UpdateLockUi(session);
            }
            else
            {
                MessageBox.Show("密码错误。", "管理员解锁", MessageBoxButton.OK, MessageBoxImage.Warning);
            }
        }
    }

    private async void Save_Click(object sender, RoutedEventArgs e)
    {
        var session = Vm?.Session;
        if (session is null) return;
        if (session.IsDeploy && !session.AdminUnlocked)
        {
            MessageBox.Show("请先解锁后再保存。", "设置", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }

        try
        {
            var c = session.Config;
            c.PtPath = PtPathBox.Text.Trim();
            c.OnnxPath = OnnxPathBox.Text.Trim();
            c.UseGpu = UseGpuBox.IsChecked == true;
            c.EnableLocalDiagnostics = LocalDiagBox.IsChecked == true;
            c.YoloPath = YoloPathBox.Text.Trim();
            c.SahiDevice = SahiDeviceBox.Text.Trim();
            c.SahiSliceSize = ParseInt(SliceSizeBox.Text, c.SahiSliceSize);
            c.SahiOverlap = ParseDouble(OverlapBox.Text, c.SahiOverlap);
            c.SahiDetConf = ParseDouble(DetConfBox.Text, c.SahiDetConf);
            c.SahiBatchSize = ParseInt(BatchSizeBox.Text, c.SahiBatchSize);
            c.SahiCropPadding = ParseInt(PaddingBox.Text, c.SahiCropPadding);
            c.SahiOutputDir = OutputDirBox.Text.Trim();
            c.SahiIosThresh = ParseDouble(IosBox.Text, c.SahiIosThresh);
            c.SahiMinAreaRatio = ParseDouble(MinAreaBox.Text, c.SahiMinAreaRatio);
            c.SahiMaxAspectRatio = ParseDouble(AspectBox.Text, c.SahiMaxAspectRatio);
            c.SahiEdgeFilter = EdgeFilterBox.IsChecked == true;
            c.SahiEdgeMarginPx = ParseInt(EdgeMarginBox.Text, c.SahiEdgeMarginPx);

            session.ConfigService.Save(c);
            session.Config = c;
            LocalDiagnostics.Configure(session.AppRoot, c.EnableLocalDiagnostics);
            MessageBox.Show("配置已保存。", "设置", MessageBoxButton.OK, MessageBoxImage.Information);
            if (Vm != null)
                await Vm.InitializeAsync();
        }
        catch (Exception ex)
        {
            MessageBox.Show("保存失败: " + ex.Message, "设置", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private async void Reload_Click(object sender, RoutedEventArgs e)
    {
        if (Vm is null) return;
        await Vm.InitializeAsync();
        RefreshFromSession();
    }

    private static int ParseInt(string text, int fallback)
        => int.TryParse(text.Trim(), out var v) ? v : fallback;

    private static double ParseDouble(string text, double fallback)
        => double.TryParse(text.Trim(), out var v) ? v : fallback;
}
