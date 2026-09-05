# Admin restart of Cognitive Router service — loads the freshly compiled dist/server.js.
# Run elevated (the service runs as LocalSystem; a normal shell cannot stop it).
# Usage: right-click -> Run with PowerShell (admin), or:
#   Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File <this script>'
Stop-Service -Name 'cognitiverouter.exe' -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
Get-NetTCPConnection -LocalPort 3456 -State Listen -ErrorAction SilentlyContinue | ForEach-Object {
    Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 2
Start-Service -Name 'cognitiverouter.exe'
Start-Sleep -Seconds 8
$conn = Get-NetTCPConnection -LocalPort 3456 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($conn) { Write-Output "Port 3456: PID $($conn.OwningProcess)" } else { Write-Output "WARNING: nothing listening on 3456" }
try { (Invoke-WebRequest -Uri 'http://127.0.0.1:3456/health' -UseBasicParsing -TimeoutSec 20).Content.Substring(0,60) } catch { Write-Output "health check failed: $($_.Exception.Message)" }
