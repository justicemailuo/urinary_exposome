$ErrorActionPreference = "Stop"

$ProjectWindows = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$DriveLetter = [System.IO.Path]::GetPathRoot($ProjectWindows).Substring(0, 1).ToLowerInvariant()
$RelativeProject = $ProjectWindows.Substring(3).Replace("\", "/")
$ProjectWsl = "/mnt/$DriveLetter/$RelativeProject"
if (-not $ProjectWsl) {
  throw "Could not convert the project path for WSL: $ProjectWindows"
}

Write-Host "Starting vLLM from $ProjectWsl"
wsl.exe -d Ubuntu-22.04 -- bash -lc "cd '$ProjectWsl' && bash scripts/start_wsl_vllm_qwen25_7b.sh"
