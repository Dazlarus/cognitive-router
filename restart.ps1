# Restart Cognitive Router without UAC prompts
# Uses a scheduled task (created once with elevation) to restart the service
$taskName = "RestartCognitiveRouter"

# Check if task exists
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue

if (-not $task) {
    Write-Host "Creating scheduled task (one-time UAC prompt)..."
    
    # Create the action
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-WindowStyle Hidden -Command `"Restart-Service -Name 'Cognitive Router' -Force`""
    
    # Create the principal (run as SYSTEM, highest privileges)
    $principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -RunLevel Highest
    
    # Register the task (this is the only time UAC appears)
    Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Force
    
    Write-Host "Task created. Future restarts will NOT trigger UAC."
}

# Run the task (no UAC!)
Write-Host "Restarting Cognitive Router..."
Start-ScheduledTask -TaskName $taskName

# Wait for completion
Start-Sleep -Seconds 3

# Check health
for ($i = 0; $i -lt 10; $i++) {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:3456/health" -TimeoutSec 5
        if ($health.status -eq "ok") {
            Write-Host "[OK] Cognitive Router is healthy"
            exit 0
        }
    } catch {
        Start-Sleep -Seconds 2
    }
}

Write-Host "[WARN] Service didn't come up in time -- check logs"
exit 1
