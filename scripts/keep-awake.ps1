# Keeps this process (and therefore the window that dot-sources this file)
# from letting Windows go into standby. No admin rights needed; display can
# still turn off. Effect lasts only for this process's lifetime — closing
# the window / killing the process re-allows standby automatically.
try {
  Add-Type -Namespace Jarvis -Name Power -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@ -ErrorAction Stop
  # ES_CONTINUOUS (0x80000000) | ES_SYSTEM_REQUIRED (0x00000001). Written
  # as decimal, not hex: PowerShell hex/octal/binary integer literals are
  # typed by raw bit pattern, so 0x80000001 parses as the Int32 -2147483647
  # rather than promoting to Int64 — casting that negative value to uint32
  # then fails a checked range conversion. Decimal literals don't have that
  # quirk and promote normally once they exceed Int32.MaxValue.
  $prev = [Jarvis.Power]::SetThreadExecutionState([uint32]2147483649)
  if ($prev -eq 0) {
    Write-Host "warn: SetThreadExecutionState failed (standby prevention not active)" -ForegroundColor Yellow
  }
} catch {
  Write-Host "warn: could not disable standby: $($_.Exception.Message)" -ForegroundColor Yellow
}
