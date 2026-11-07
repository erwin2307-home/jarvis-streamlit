# Starts a Cloudflare quick tunnel for the remote API and actively watches
# it, not just whether the cloudflared process is still running.
#
# Why this exists: a quick tunnel (`cloudflared tunnel --url ...`) can end
# up ALIVE but non-functional — stuck retrying "Unauthorized: Tunnel not
# found" against Cloudflare's edge forever without ever exiting. A restart-
# on-exit loop (what this used to be) never notices that, so the printed
# URL silently goes dead (DNS stops resolving) while the window looks fine.
# This script polls the tunnel's own /health through the public URL on an
# interval and force-restarts cloudflared if that stops answering, even
# though the process is technically still alive.
param(
  [Parameter(Mandatory = $true)][string]$CloudflaredBin,
  [Parameter(Mandatory = $true)][int]$Port,
  [Parameter(Mandatory = $true)][string]$ApiKey,
  [Parameter(Mandatory = $true)][string]$LogFile,
  [Parameter(Mandatory = $true)][string]$SecretsFile
)

function Get-TunnelUrlFromLog {
  if (Test-Path $LogFile) {
    $m = Select-String -Path $LogFile -Pattern "https://[a-zA-Z0-9-]+\.trycloudflare\.com" -ErrorAction SilentlyContinue |
      Select-Object -Last 1
    if ($m) { return $m.Matches[0].Value }
  }
  return $null
}

function Announce-Secrets {
  param([string]$Url)
  $block = "API_URL = `"$Url`"`nAPI_KEY = `"$ApiKey`""

  $writeOk = $true
  $writeErr = $null
  try {
    Set-Content -Path $SecretsFile -Value $block -Encoding utf8 -ErrorAction Stop
  } catch {
    $writeOk = $false
    $writeErr = $_.Exception.Message
  }

  $copied = $false
  $copyErr = $null
  try { Set-Clipboard -Value $block -ErrorAction Stop; $copied = $true } catch { $copyErr = $_.Exception.Message }

  # Status marker has no secret content — safe to inspect from outside
  # this window (e.g. to confirm the write succeeded) without exposing
  # the API key.
  $statusFile = "$SecretsFile.status"
  "at=$(Get-Date -Format o); file_write=$writeOk; write_err=$writeErr; clipboard=$copied; clipboard_err=$copyErr" |
    Set-Content -Path $statusFile -Encoding utf8

  if (-not $writeOk) {
    Write-Host "[watchdog] konnte Secrets-Datei nicht schreiben: $writeErr" -ForegroundColor Red
  }

  Write-Host ""
  Write-Host "=== Tunnel aktiv: $Url ===" -ForegroundColor Cyan
  Write-Host "-----------------------------------------------------------"
  Write-Host $block
  Write-Host "-----------------------------------------------------------"
  if ($copied) {
    Write-Host "(in Zwischenablage kopiert - direkt einfuegen)" -ForegroundColor Green
  } else {
    Write-Host "(Zwischenablage-Kopie fehlgeschlagen - Datei zum Kopieren: $SecretsFile)" -ForegroundColor Yellow
  }
  Write-Host ""
}

while ($true) {
  Remove-Item $LogFile -ErrorAction SilentlyContinue
  Write-Host "[watchdog] starte cloudflared..." -ForegroundColor Yellow
  $proc = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-Command", "& '$CloudflaredBin' tunnel --url http://localhost:$Port *>> '$LogFile'") `
    -PassThru -WindowStyle Hidden

  # --- wait for the URL to show up in the log ---
  $url = $null
  $deadline = (Get-Date).AddSeconds(30)
  while ((Get-Date) -lt $deadline -and -not $url) {
    Start-Sleep -Milliseconds 800
    $url = Get-TunnelUrlFromLog
  }
  if (-not $url) {
    Write-Host "[watchdog] keine Tunnel-URL nach 30s erhalten - Neustart" -ForegroundColor Red
    try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch { }
    Start-Sleep -Seconds 3
    continue
  }

  Announce-Secrets -Url $url

  # --- health-watch this URL until it (or the process) dies ---
  $fails = 0
  $healthy = $true
  while ($healthy) {
    Start-Sleep -Seconds 15
    if ($proc.HasExited) {
      Write-Host "[watchdog] cloudflared-Prozess beendet - Neustart" -ForegroundColor Red
      $healthy = $false
      break
    }
    try {
      $resp = Invoke-WebRequest -Uri "$url/health" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
      if ($resp.StatusCode -eq 200) {
        $fails = 0
      } else {
        $fails++
      }
    } catch {
      $fails++
    }
    if ($fails -ge 3) {
      Write-Host "[watchdog] Tunnel antwortet seit ~45s nicht mehr - Neustart" -ForegroundColor Red
      try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch { }
      $healthy = $false
    }
  }
  Start-Sleep -Seconds 2
}
