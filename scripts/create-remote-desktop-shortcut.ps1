# Erstellt ein "Start Jarvis Remote"-Shortcut auf dem Desktop.
# Aufruf: powershell -ExecutionPolicy Bypass -File scripts\create-remote-desktop-shortcut.ps1

$ScriptPath = $MyInvocation.MyCommand.Path
$ScriptDir = Split-Path -Parent $ScriptPath
$RepoRoot = Split-Path -Parent $ScriptDir
$Launcher = Join-Path $RepoRoot "scripts\start-streamlit-remote.ps1"
$IconSrc = Join-Path $RepoRoot "frontend\src-tauri\icons\icon.ico"
$Desktop = [Environment]::GetFolderPath("Desktop")
$LnkPath = Join-Path $Desktop "Start Jarvis Remote.lnk"

Write-Host "RepoRoot : $RepoRoot"
Write-Host "Launcher : $Launcher"
Write-Host "Shortcut : $LnkPath"
Write-Host ""

$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut($LnkPath)
$lnk.TargetPath = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$lnk.Arguments = "-NoExit -NoProfile -ExecutionPolicy Bypass -File `"$Launcher`""
$lnk.WorkingDirectory = $RepoRoot
$lnk.WindowStyle = 1
$lnk.Description = "Start Ollama + remote chat API + Cloudflare tunnel for the Streamlit frontend"
if (Test-Path $IconSrc) { $lnk.IconLocation = "$IconSrc,0" }
$lnk.Save()

Write-Host "Shortcut gespeichert."
