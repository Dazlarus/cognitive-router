# Cognitive Router launcher script
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Install dependencies if needed
if (!(Test-Path (Join-Path $ScriptDir "node_modules"))) {
    Write-Host "Installing dependencies..."
    Push-Location $ScriptDir
    npm install
    Pop-Location
}

# Build TypeScript
Write-Host "Compiling TypeScript..."
Push-Location $ScriptDir
npx tsc
Pop-Location

Write-Host "Starting Cognitive Router..."
Push-Location $ScriptDir
node dist/server.js
Pop-Location
