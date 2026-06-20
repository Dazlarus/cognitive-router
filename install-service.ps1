# Install Cognitive Router as a Windows Service
# Requires: nssm.exe (download from https://nssm.cc/download)
# Run this as Administrator

$ErrorActionPreference = "Stop"

# Configuration
$ServiceName = "CognitiveRouter"
$DisplayName = "Cognitive Router Self-Healing Model Proxy"
$ServiceDescription = "Self-healing model routing proxy. Owns the entire failure lifecycle — tries models, retries on failure, circuit breakers, and returns clean responses. OpenAI-compatible endpoint on port 3456."

# Paths — resolve relative to this script
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = $ScriptDir
$EnvFile = Join-Path $ProjectDir ".env"
$NssmPath = "C:\nssm\nssm.exe"  # Change this if NSSM is in a different location

# Resolve Node.js path
$NodeExe = (Get-Command node -ErrorAction Stop).Source

# Check if NSSM is available
if (!(Test-Path $NssmPath)) {
    Write-Error "NSSM not found at $NssmPath. Download from https://nssm.cc/download and install it first."
    exit 1
}

# Check if .env exists
if (!(Test-Path $EnvFile)) {
    Write-Error "Environment file not found at $EnvFile. Create .env with your API keys first (see .env.example)."
    exit 1
}

# Create logs directory
$LogDir = "C:\Logs\CognitiveRouter"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host "=== Installing Cognitive Router as a Windows Service ===" -ForegroundColor Cyan
Write-Host "Service name: $ServiceName"
Write-Host "Display name: $DisplayName"
Write-Host "Description: $ServiceDescription"
Write-Host "Node.js:      $NodeExe"
Write-Host "Project dir:  $ProjectDir"
Write-Host "Log directory: $LogDir"
Write-Host ""

# Remove existing service if present
Write-Host "Removing existing service if present..."
& $NssmPath remove $ServiceName confirm 2>$null

# Create the service
Write-Host "Creating service..."
& $NssmPath install $ServiceName $NodeExe "$ProjectDir\dist\server.js"

# Configure service to start automatically
Write-Host "Configuring service to start automatically..."
& $NssmPath set $ServiceName Start SERVICE_AUTO_START

# Configure working directory
Write-Host "Setting working directory..."
& $NssmPath set $ServiceName AppDirectory $ProjectDir

# Configure environment variables from .env
Write-Host "Loading environment variables from .env..."
Get-Content $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and !$line.StartsWith("#") -and $line -match "^([^=]+)=(.*)$") {
        $key = $matches[1].Trim()
        $val = $matches[2].Trim()
        & $NssmPath set $ServiceName AppEnvironmentExtra "$key=$val" 2>$null
    }
}

# Configure log files
Write-Host "Configuring log files..."
& $NssmPath set $ServiceName AppStdout "$LogDir\stdout.log"
& $NssmPath set $ServiceName AppStderr "$LogDir\stderr.log"

# Configure restart behavior
& $NssmPath set $ServiceName AppExit Default Restart
& $NssmPath set $ServiceName AppRestartDelay 10000

Write-Host ""
Write-Host "✅ Service installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "To start the service manually, run:"
Write-Host "  sc start $ServiceName"
Write-Host ""
Write-Host "To view logs:"
Write-Host "  Get-Content '$LogDir\stdout.log' -Wait"
Write-Host ""
Write-Host "To uninstall, run:"
Write-Host "  & '$NssmPath' remove $ServiceName confirm"
