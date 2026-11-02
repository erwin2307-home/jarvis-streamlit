# One-click Jarvis launcher.
# Starts Ollama (if needed), backend pipeline, and the Tauri frontend in
# separate PowerShell windows. Waits for the backend WebSocket port to
# come up before launching the UI.

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BackendDir = Join-Path $RepoRoot "backend"
$FrontendDir = Join-Path $RepoRoot "frontend"

# --- Resolve uv (winget install puts it outside PATH until shell restart) ---
$UvBin = Get-Command uv -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
if (-not $UvBin) {
  $candidates = @(
    "$env:USERPROFILE\.local\bin\uv.exe",
    "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe"
  ) + (Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv_*" -Filter "uv.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
  foreach ($c in $candidates) {
    if ($c -and (Test-Path $c)) { $UvBin = $c; break }
  }
}
if (-not $UvBin) {
  Write-Host "uv not found. Install: winget install astral-sh.uv" -ForegroundColor Red
  exit 1
}

# --- Resolve ollama binary ---
$OllamaBin = Get-Command ollama -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
if (-not $OllamaBin) {
  $candidate = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
  if (Test-Path $candidate) { $OllamaBin = $candidate }
}
if (-not $OllamaBin) {
  Write-Host "ollama not found. Install: winget install Ollama.Ollama" -ForegroundColor Red
  exit 1
}

# --- 1) Start Ollama service if not already running ---
$ollamaUp = $false
try {
  $resp = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/version" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
  if ($resp.StatusCode -eq 200) { $ollamaUp = $true }
} catch { }

if ($ollamaUp) {
  Write-Host "ok: ollama already serving on :11434"
} else {
  Write-Host "starting ollama serve..." -ForegroundColor Yellow
  # Low-VRAM GPUs (e.g. GeForce 940MX, 2GB) crash llama runner on GPU
  # offload ("failed to allocate compute pp buffers"). Force CPU backend.
  $env:OLLAMA_LLM_LIBRARY = "cpu"
  Start-Process -FilePath $OllamaBin -ArgumentList "serve" -WindowStyle Hidden
  # Wait up to 15s for the service to come up.
  $deadline = (Get-Date).AddSeconds(15)
  while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 500
    try {
      $resp = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/version" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
      if ($resp.StatusCode -eq 200) { $ollamaUp = $true; break }
    } catch { }
  }
  if (-not $ollamaUp) {
    Write-Host "warn: ollama service did not respond within 15 s (continuing anyway)" -ForegroundColor Yellow
  } else {
    Write-Host "ok: ollama up"
  }
}

# --- 2) Backend in its own window ---
$backendCmd = @"
`$env:JARVIS_START_ORCHESTRATOR='1'
`$env:JARVIS_SILENCE_MS='500'
`$env:JARVIS_USE_CLOUD='0'
Set-Location '$BackendDir'
& '$UvBin' run uvicorn jarvis.app:app --port 8765
"@
Write-Host "starting backend (uvicorn on :8765)..."
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", $backendCmd

# --- 3) Wait for backend to be ready on :8765 ---
Write-Host "waiting for backend health..." -NoNewline
$deadline = (Get-Date).AddSeconds(120)
$backendUp = $false
while ((Get-Date) -lt $deadline) {
  Start-Sleep -Milliseconds 800
  Write-Host "." -NoNewline
  try {
    $resp = Invoke-WebRequest -Uri "http://127.0.0.1:8765/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
    if ($resp.StatusCode -eq 200) { $backendUp = $true; break }
  } catch { }
}
Write-Host ""

if (-not $backendUp) {
  Write-Host "error: backend not reachable after 120 s - check the backend window" -ForegroundColor Red
  exit 2
}
Write-Host "ok: backend ready on :8765" -ForegroundColor Green

# --- 4) Frontend (Tauri) in its own window ---
$frontendCmd = @"
Set-Location '$FrontendDir'
npm run tauri dev
"@
Write-Host "starting frontend (npm run tauri dev)..."
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", $frontendCmd

Write-Host ""
Write-Host "Jarvis launched. Say 'Jarvis' (or press Win+J / click HUD) to start a turn." -ForegroundColor Cyan
Write-Host "Close the backend / frontend windows to stop everything."
