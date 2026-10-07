[Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if ($engine) {
    Write-Output "Windows OCR Engine created successfully: $($engine.RecognizerLanguage.DisplayName)"
} else {
    Write-Output "Windows OCR not available via user profile"
}
