# Simple DeepSeek Monitoring Script (PowerShell)
# Run this script to continuously monitor DeepSeek status

Write-Host "DEEPSEEK MONITORING - Simple Check" -ForegroundColor Cyan
Write-Host "===================================`n" -ForegroundColor Cyan

while ($true) {
    Write-Host "`n[$(Get-Date -Format 'HH:mm:ss')] Running DeepSeek status check..." -ForegroundColor Yellow
    
    # Run the Python monitoring script
    & ".\\.venv\Scripts\python.exe" monitor_deepseek.py
    
    Write-Host "`nWaiting 60 seconds before next check..." -ForegroundColor Gray
    Write-Host "(Press Ctrl+C to stop)`n" -ForegroundColor Gray
    
    Start-Sleep -Seconds 60
}

