# Streamlit Runner with Log File
# Add Python Scripts to PATH
$env:Path += ";C:\Users\Monir\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts"

# Set DeepSeek API Key
$env:DEEPSEEK_API_KEY = "sk-64543b775f5546f0ba0e313366a1b550"

# Find Python with LangChain - try py launcher first (it found LangChain)
$logFile = "streamlit_logs.txt"
$appPath = "ui\chat_demo.py"

# Check if py launcher has LangChain
try {
    $pyCheck = py -c "import langchain; print('OK')" 2>&1
    if ($LASTEXITCODE -eq 0) {
        $usePy = $true
        Write-Host "Found LangChain via py launcher" -ForegroundColor Green
    } else {
        $usePy = $false
    }
} catch {
    $usePy = $false
}

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Sales Intelligence Agentic System - Streamlit App" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting Streamlit application..." -ForegroundColor Yellow
Write-Host "Logs will be saved to: $logFile" -ForegroundColor Cyan
Write-Host "App URL: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Run streamlit using the Python that has LangChain
if ($usePy) {
    Write-Host "Using 'py' launcher (has LangChain installed)" -ForegroundColor Green
    Write-Host ""
    # Use py launcher to run streamlit
    py -m streamlit run $appPath 2>&1 | Tee-Object -FilePath $logFile
} else {
    Write-Host "Using streamlit from PATH..." -ForegroundColor Yellow
    Write-Host ""
    # Fallback to streamlit from PATH
    streamlit run $appPath 2>&1 | Tee-Object -FilePath $logFile
}

