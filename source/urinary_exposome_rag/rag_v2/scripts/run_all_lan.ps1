$ErrorActionPreference = "Stop"

$Project = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ApiLog = Join-Path $Project "fastapi_v2.log"
$StreamlitLog = Join-Path $Project "streamlit_v2.log"

Set-Location $Project

foreach ($port in 8890, 8501) {
  Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | ForEach-Object {
    Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
  }
}

# Keep the FastAPI backend private to this machine. Only Streamlit is exposed on the LAN.
Start-Process -FilePath powershell -ArgumentList @(
  "-NoExit",
  "-ExecutionPolicy",
  "Bypass",
  "-Command",
  "cd '$Project'; python -m uvicorn backend.main:app --host 127.0.0.1 --port 8890 *> '$ApiLog'"
) -WindowStyle Hidden

Start-Sleep -Seconds 6

Start-Process -FilePath powershell -ArgumentList @(
  "-NoExit",
  "-ExecutionPolicy",
  "Bypass",
  "-Command",
  "cd '$Project'; `$env:RAG_API_BASE_URL='http://127.0.0.1:8890'; `$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS='false'; python -m streamlit run .\streamlit_app\app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true --browser.gatherUsageStats false *> '$StreamlitLog'"
) -WindowStyle Hidden

Start-Sleep -Seconds 6

$ip = Get-NetIPAddress -AddressFamily IPv4 |
  Where-Object { $_.IPAddress -notlike "127.*" -and $_.PrefixOrigin -ne "WellKnown" } |
  Select-Object -First 1 -ExpandProperty IPAddress

Write-Host "LAN Streamlit: http://$ip`:8501"
Write-Host "Local FastAPI: http://127.0.0.1:8890"
Write-Host "If other devices cannot open it, allow TCP 8501 in Windows Firewall."
Get-NetTCPConnection -LocalPort 8890,8501 -State Listen -ErrorAction SilentlyContinue |
  Select-Object LocalAddress, LocalPort, OwningProcess
