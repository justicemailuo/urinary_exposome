$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Project
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8890 --reload
