$ErrorActionPreference = "SilentlyContinue"

foreach ($port in 8890, 8501) {
  Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | ForEach-Object {
    Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
  }
}

Write-Host "Stopped FastAPI/Streamlit services on ports 8890 and 8501."
