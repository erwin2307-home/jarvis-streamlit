# One-click launcher for the Streamlit-remote setup.
# Starts Ollama (if needed), the remote chat API, and a Cloudflare quick
# tunnel in separate windows. Prints (and copies to clipboard) the
# tunnel URL + API key to paste into the Streamlit app's sidebar.

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BackendDir = Join-Path $RepoRoot "backend"
$KeyFile = Join-Path $RepoRoot "remote_api_key.txt"

if (-not (Test-Path $KeyFile)) {
  Write-Host "error: $KeyFile not found. Generate one first (see STREAMLIT_SETUP.md)." -ForegroundColor Red
  exit 1
}
$ApiKey = (Get-Content $KeyFile -Raw).Trim()

# --- Resolve uv ---
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

# --- Resolve ollama ---
$OllamaBin = Get-Command ollama -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
if (-not $OllamaBin) {
  $candidate = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
  if (Test-Path $candidate) { $OllamaBin = $candidate }
}
if (-not $OllamaBin) {
  Write-Host "ollama not found. Install: winget install Ollama.Ollama" -ForegroundColor Red
  exit 1
}

# --- Resolve cloudflared ---
$CloudflaredBin = Get-Command cloudflared -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
if (-not $CloudflaredBin) {
  $candidate = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
  if (Test-Path $candidate) { $CloudflaredBin = $candidate }
}
if (-not $CloudflaredBin) {
  Write-Host "cloudflared not found. Install: winget install Cloudflare.cloudflared" -ForegroundColor Red
  exit 1
}

# --- 1) Ollama ---
$ollamaUp = $false
try {
  $resp = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/version" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
  if ($resp.StatusCode -eq 200) { $ollamaUp = $true }
} catch { }

if ($ollamaUp) {
  Write-Host "ok: ollama already serving on :11434"
} else {
  Write-Host "starting ollama serve..." -ForegroundColor Yellow
  $env:OLLAMA_LLM_LIBRARY = "cpu"
  Start-Process -FilePath $OllamaBin -ArgumentList "serve" -WindowStyle Hidden
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

# --- 2) Remote chat API in its own window ---
$apiCmd = @"
`$env:JARVIS_REMOTE_API_KEY='$ApiKey'
Set-Location '$BackendDir'
& '$UvBin' run uvicorn jarvis.remote_api:app --host 0.0.0.0 --port 8800
"@
Write-Host "starting remote chat API (uvicorn on :8800)..."
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", $apiCmd

Write-Host "waiting for remote API health..." -NoNewline
$deadline = (Get-Date).AddSeconds(60)
$apiUp = $false
while ((Get-Date) -lt $deadline) {
  Start-Sleep -Milliseconds 800
  Write-Host "." -NoNewline
  try {
    $resp = Invoke-WebRequest -Uri "http://127.0.0.1:8800/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
    if ($resp.StatusCode -eq 200) { $apiUp = $true; break }
  } catch { }
}
Write-Host ""
if (-not $apiUp) {
  Write-Host "error: remote API not reachable after 60 s - check its window" -ForegroundColor Red
  exit 2
}
Write-Host "ok: remote API ready on :8800" -ForegroundColor Green

# --- 3) Cloudflare quick tunnel ---
$TunnelLog = Join-Path $env:TEMP "jarvis-cloudflared.log"
$TunnelOutLog = Join-Path $env:TEMP "jarvis-cloudflared.out.log"
Remove-Item $TunnelLog, $TunnelOutLog -ErrorAction SilentlyContinue
Write-Host "starting cloudflare tunnel..."
Start-Process -FilePath $CloudflaredBin -ArgumentList "tunnel", "--url", "http://localhost:8800" `
  -WindowStyle Hidden -RedirectStandardError $TunnelLog -RedirectStandardOutput $TunnelOutLog

Write-Host "waiting for tunnel URL..." -NoNewline
$deadline = (Get-Date).AddSeconds(30)
$TunnelUrl = $null
while ((Get-Date) -lt $deadline) {
  Start-Sleep -Milliseconds 800
  Write-Host "." -NoNewline
  if (Test-Path $TunnelLog) {
    $m = Select-String -Path $TunnelLog -Pattern "https://[a-zA-Z0-9-]+\.trycloudflare\.com" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($m) { $TunnelUrl = $m.Matches[0].Value; break }
  }
}
Write-Host ""

if (-not $TunnelUrl) {
  Write-Host "error: tunnel URL not found after 30 s - check $TunnelLog" -ForegroundColor Red
  exit 3
}

try { Set-Clipboard -Value $TunnelUrl } catch { }

Write-Host ""
Write-Host "=== Jarvis Remote ist bereit ===" -ForegroundColor Cyan
Write-Host "Tunnel-URL (in Zwischenablage kopiert): $TunnelUrl" -ForegroundColor Green
Write-Host "API-Key: $ApiKey"
Write-Host ""
Write-Host "In der Streamlit-App (Sidebar) eintragen:"
Write-Host "  API-URL = $TunnelUrl"
Write-Host "  API-Key = $ApiKey"
Write-Host ""
Write-Host "Fenster offen lassen, solange die Streamlit-App erreichbar sein soll."
