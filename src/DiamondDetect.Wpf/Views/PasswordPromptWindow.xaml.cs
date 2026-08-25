using System.Windows;

namespace DiamondDetect.Wpf.Views;

public partial class PasswordPromptWindow : Window
{
    public PasswordPromptWindow()
    {
        InitializeComponent();
        PasswordBox.Focus();
    }

    public string Password => PasswordBox.Password;

    private void Ok_Click(object sender, RoutedEventArgs e)
    {
        DialogResult = true;
        Close();
    }

    private void Cancel_Click(object sender, RoutedEventArgs e)
    {
        DialogResult = false;
        Close();
    }
}
