# Streamlit launcher for Sales Intelligence Agentic System
param(
    [string]$DeepSeekKey
)

$ErrorActionPreference = "Stop"
$defaultDeepSeekKey = "sk-64543b775f5546f0ba0e313366a1b550"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "=== Sales Intelligence Agentic System ===" -ForegroundColor Cyan

# Ensure virtual environment exists
$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Yellow
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -m venv .venv
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        python -m venv .venv
    } else {
        throw "Python was not found on PATH. Install Python 3.10+ and rerun this script."
    }
}

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment creation failed."
}

Write-Host "Using Python from: $venvPython" -ForegroundColor Green

# Upgrade pip and install requirements
Write-Host "Installing/Checking dependencies..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip | Out-Host
& $venvPython -m pip install -r requirements.txt | Out-Host

# Resolve DeepSeek API key (prefer argument, then env, then default)
if (-not $DeepSeekKey) {
    $DeepSeekKey = $env:DEEPSEEK_API_KEY
}
if (-not $DeepSeekKey) {
    $DeepSeekKey = $defaultDeepSeekKey
}

if ($DeepSeekKey) {
    $env:DEEPSEEK_API_KEY = $DeepSeekKey
    Write-Host "DeepSeek API key configured for this session." -ForegroundColor Green
} else {
    throw "DeepSeek API key is required; set DEEPSEEK_API_KEY environment variable or rerun with -DeepSeekKey."
}

# Launch Streamlit
Write-Host "Starting Streamlit app..." -ForegroundColor Cyan
Write-Host "Hold Ctrl+C to stop." -ForegroundColor Yellow
Write-Host "URL: http://localhost:8501" -ForegroundColor Magenta
Write-Host ""

& $venvPython -m streamlit run ui/chat_demo.py

