using System.IO;
using System.Windows;

namespace DiamondDetect.Wpf.Views;

public partial class InputPromptWindow : Window
{
    public InputPromptWindow(string title, string message, string defaultValue)
    {
        InitializeComponent();
        Title = title;
        MessageText.Text = message;
        InputBox.Text = defaultValue;
        InputBox.SelectAll();
        InputBox.Focus();
    }

    public string Value => InputBox.Text.Trim();

    private void Ok_Click(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrWhiteSpace(InputBox.Text))
        {
            MessageBox.Show("名称不能为空。", "提示", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        if (Value.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
        {
            MessageBox.Show("名称包含非法字符。", "提示", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        DialogResult = true;
        Close();
    }

    private void Cancel_Click(object sender, RoutedEventArgs e)
    {
        DialogResult = false;
        Close();
    }
}
