# Ensure Cognitive Router is running detached and healthy. Idempotent: exits 0 if already healthy.
# Used by scheduled task 'CognitiveRouterEnsure' (logon trigger) and by the 04:03 cron restart job (on-demand trigger).
# If a Windows-service instance ever owns port 3456 and is healthy, this script defers to it (no-ops).

$repo = 'C:\Users\hi100\.openclaw\workspace\cognitive-router'

# 1. Already healthy? Done.
try {
    $h = Invoke-RestMethod 'http://127.0.0.1:3456/health' -TimeoutSec 5
    if ($h.status -eq 'ok') { Write-Output 'healthy'; exit 0 }
} catch {}

# 2. Stale/unhealthy listener? Kill it (same-user processes only).
$l = Get-NetTCPConnection -LocalPort 3456 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($l) {
    Stop-Process -Id $l.OwningProcess -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

# 3. Build if dist is stale.
$newestSrc = (Get-ChildItem "$repo\src" -Recurse -Filter *.ts | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime
if ($newestSrc -gt (Get-Item "$repo\dist\server.js").LastWriteTime) {
    Push-Location $repo
    npm run build 2>&1 | Out-Null
    Pop-Location
}

# 4. Detached start with log capture (NEVER npm/shell-wrapper: those die with the parent session).
Start-Process -FilePath 'node' -ArgumentList 'dist/server.js' -WorkingDirectory $repo -WindowStyle Hidden `
    -RedirectStandardOutput "$repo\data\router-stdout.log" -RedirectStandardError "$repo\data\router-stderr.log" | Out-Null

# 5. Verify.
foreach ($i in 1..10) {
    Start-Sleep -Seconds 3
    try {
        $h = Invoke-RestMethod 'http://127.0.0.1:3456/health' -TimeoutSec 5
        if ($h.status -eq 'ok') { Write-Output 'started healthy'; exit 0 }
    } catch {}
}
Write-Output 'FAILED to bring router up'
exit 1
