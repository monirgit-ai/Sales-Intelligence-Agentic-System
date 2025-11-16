# Streamlit Runner Script with Logging
$env:Path += ";C:\Users\Monir\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts"

Write-Host "Starting Streamlit app..." -ForegroundColor Green
Write-Host "App will be available at: http://localhost:8501" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

# Run streamlit
streamlit run ui/chat_demo.py

