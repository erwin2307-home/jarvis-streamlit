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

# --- 0) Clean up stale processes from previous runs ---
# This script used to always open a NEW uvicorn/cloudflared window with no
# check for ones already running (unlike the Ollama step below) — re-running
# it piled up zombies across sessions instead of replacing them. One such
# zombie (a cloudflared quick-tunnel loop stuck retrying registration
# forever, alive but never serving) is exactly what caused a stale,
# unresolvable tunnel URL to sit in a pasted Streamlit secret: a fresh
# run's URL extraction could pick up a dead zombie's line from the shared
# log file instead of the new process actually just started.
Write-Host "cleaning up stale processes from previous runs..."
Get-CimInstance Win32_Process -Filter "Name='cloudflared.exe'" -ErrorAction SilentlyContinue |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -and ($_.CommandLine -match 'jarvis\.remote_api:app|tunnel-watchdog\.ps1|jarvis-cloudflared\.log') } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 1

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
# Wrapped in a restart loop: this window is the only thing keeping the API
# alive (it's a plain console process, not a service), so a crash — or the
# "low memory, background task killed" situation seen earlier with the 32b
# model pull — would otherwise silently take the whole remote setup down
# until someone notices and manually restarts this script.
#
# Also keeps the machine out of standby for as long as this window lives:
# Windows idle-sleep would otherwise kill Ollama/uvicorn/cloudflared mid-tunnel,
# breaking the Streamlit Cloud frontend with no local error to look at. Uses
# SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) — no admin
# rights needed, display is still allowed to turn off, and the flag is only
# in effect for this process's lifetime (reset automatically if the window
# is closed or the process dies), so closing the window to stop everything
# (as the final printed instructions already say) also re-allows standby.
$apiCmd = @"
. '$PSScriptRoot\keep-awake.ps1'
`$env:JARVIS_REMOTE_API_KEY='$ApiKey'
Set-Location '$BackendDir'
while (`$true) {
  & '$UvBin' run uvicorn jarvis.remote_api:app --host 0.0.0.0 --port 8800
  Write-Host ''
  Write-Host 'remote API exited - restarting in 3s (Ctrl+C to stop for good)' -ForegroundColor Yellow
  Start-Sleep -Seconds 3
}
"@
Write-Host "starting remote chat API (uvicorn on :8800, auto-restarts on crash)..."
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

# --- 3) Cloudflare quick tunnel, actively monitored ---
# A quick tunnel (no Cloudflare account, random *.trycloudflare.com host)
# can go stuck ALIVE-BUT-DEAD: cloudflared keeps running but loops
# "Unauthorized: Tunnel not found" against Cloudflare's edge forever
# without ever exiting — a plain restart-on-crash loop never notices that,
# so the already-pasted Streamlit secret silently goes stale (DNS stops
# resolving) with nothing wrong-looking in the window. tunnel-watchdog.ps1
# runs in its own visible window, polls the live URL's /health, and
# force-restarts cloudflared (re-announcing the new URL + re-copying the
# secrets) whenever that stops answering — watch that window for status.
$TunnelLog = Join-Path $env:TEMP "jarvis-cloudflared.log"
$SecretsFile = Join-Path $env:TEMP "jarvis-remote-secrets.txt"
Write-Host "starting cloudflare tunnel watchdog..."
# -Command with one script-text argument, not -File with many named
# params: the same shape already proven to work for the API window above.
# The many-element array form of -ArgumentList was tried first and never
# actually produced a surviving process here (0 cloudflared processes,
# no tunnel-watchdog.ps1 in the process list, even though Start-Process
# itself didn't throw) — whatever the exact cause, this is the pattern
# known to work, so stick to it rather than re-diagnosing a PowerShell
# argument-marshalling quirk.
$watchdogCmd = @"
& '$PSScriptRoot\tunnel-watchdog.ps1' -CloudflaredBin '$CloudflaredBin' -Port 8800 -ApiKey '$ApiKey' -LogFile '$TunnelLog' -SecretsFile '$SecretsFile'
"@
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", $watchdogCmd

Write-Host ""
Write-Host "=== Jeanny Remote gestartet ===" -ForegroundColor Cyan
Write-Host "Die Tunnel-URL + Secrets erscheinen im Watchdog-Fenster, sobald der"
Write-Host "Tunnel steht (und erneut, falls er je neu verbinden muss)."
Write-Host "-> Streamlit Cloud: App -> Settings -> Secrets -> einfuegen -> Save"
Write-Host "-> Oder direkt in der App-Sidebar bei API-URL / API-Key einfuegen"
Write-Host "Falls die Zwischenablage mal nicht geht, steht der Block zusaetzlich"
Write-Host "in: $SecretsFile"
Write-Host ""
Write-Host "Alle Fenster offen lassen, solange die App erreichbar sein soll."
Write-Host "(API und Tunnel starten bei einem Absturz/Haenger automatisch neu.)" -ForegroundColor DarkGray
