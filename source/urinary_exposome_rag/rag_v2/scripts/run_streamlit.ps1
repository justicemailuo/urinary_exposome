$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Project
if (-not $env:RAG_API_BASE_URL) {
  $env:RAG_API_BASE_URL = "http://127.0.0.1:8890"
}
python -m streamlit run .\streamlit_app\app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
