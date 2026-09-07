# Rebuild the Cognitive Router protection stack after the 2026.9.1 update wiped it.
# Idempotent-ish: kills orphans on 3456, reinstalls service, configures failure recovery,
# starts it, recreates the logon ensure-task, verifies health.
$ErrorActionPreference = 'Continue'
$repo = 'C:\Users\hi100\.openclaw\workspace\cognitive-router'
$daemon = "$repo\dist\daemon"

Write-Output '== 1. stop orphaned LocalSystem processes on 3456 =='
foreach ($procId in (Get-NetTCPConnection -LocalPort 3456 -State Listen -ErrorAction SilentlyContinue).OwningProcess | Select-Object -Unique) {
    Write-Output "killing orphan PID $procId"
    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 2

Write-Output '== 2. guard: node.exe path from xml still exists =='
if (-not (Test-Path 'C:\Program Files\nodejs\node.exe')) {
    Write-Output 'FATAL: xml node.exe path missing - fix xml before install'
    exit 1
}

Write-Output '== 3. reinstall service (node-windows wrapper) =='
& "$daemon\cognitiverouter.exe" uninstall 2>&1 | Out-Null
Start-Sleep -Seconds 2
& "$daemon\cognitiverouter.exe" install 2>&1 | Select-Object -First 5

Write-Output '== 4. start=auto + failure recovery (restart x3) =='
sc.exe config cognitiverouter.exe start= auto | Out-Null
sc.exe failure cognitiverouter.exe reset= 0 actions= restart/5000/restart/5000/restart/5000 | Out-Null

Write-Output '== 5. start service =='
sc.exe start cognitiverouter.exe | Out-Null
Start-Sleep -Seconds 6

Write-Output '== 6. recreate ensure task (logon trigger) =='
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-NoProfile -WindowStyle Hidden -File C:\Users\hi100\.openclaw\workspace\cognitive-router\scripts\ensure-router.ps1'
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
Register-ScheduledTask -TaskName 'CognitiveRouterEnsure' -Action $action -Trigger $trigger -Settings $settings -Force | Select-Object TaskName,State | Format-Table -AutoSize

Write-Output '== 7. verify =='
Get-Service | Where-Object { $_.Name -eq 'cognitiverouter.exe' } | Select-Object Name,Status,StartType | Format-Table -AutoSize
foreach ($i in 1..10) {
    Start-Sleep -Seconds 3
    try {
        $h = Invoke-RestMethod 'http://127.0.0.1:3456/health' -TimeoutSec 5
        if ($h.status -eq 'ok') { Write-Output "HEALTH: ok (attempt $i)"; exit 0 }
    } catch {}
}
Write-Output 'HEALTH: FAILED'
exit 1
