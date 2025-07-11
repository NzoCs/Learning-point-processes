# EasyTPP CLI PowerShell Wrapper
# Professional PowerShell script for Windows users

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonScript = Join-Path $ScriptDir "easytpp_cli.py"
$ConfigDir = Join-Path $ScriptDir "configs"
$OutputDir = Join-Path $ScriptDir "outputs"

# Colors for output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    
    switch ($Color) {
        "Red" { Write-Host $Message -ForegroundColor Red }
        "Green" { Write-Host $Message -ForegroundColor Green }
        "Yellow" { Write-Host $Message -ForegroundColor Yellow }
        "Blue" { Write-Host $Message -ForegroundColor Blue }
        "Cyan" { Write-Host $Message -ForegroundColor Cyan }
        "Magenta" { Write-Host $Message -ForegroundColor Magenta }
        default { Write-Host $Message }
    }
}

# Check if Python is available
function Test-PythonAvailability {
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✅ Python found: $pythonVersion" "Green"
            return $true
        }
    }
    catch {
        Write-ColorOutput "❌ Python not found in PATH" "Red"
        Write-ColorOutput "Please install Python 3.8+ and add it to your PATH" "Yellow"
        return $false
    }
    return $false
}

# Check if CLI script exists
function Test-CLIScript {
    if (Test-Path $PythonScript) {
        Write-ColorOutput "✅ EasyTPP CLI script found" "Green"
        return $true
    } else {
        Write-ColorOutput "❌ EasyTPP CLI script not found: $PythonScript" "Red"
        return $false
    }
}

# Create directories if they don't exist
function Initialize-Directories {
    $dirs = @($ConfigDir, $OutputDir, (Join-Path $ScriptDir "logs"), (Join-Path $ScriptDir "checkpoints"))
    
    foreach ($dir in $dirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-ColorOutput "📁 Created directory: $dir" "Cyan"
        }
    }
}

# Show help if no arguments
function Show-Help {
    Write-ColorOutput @"
🚀 EasyTPP CLI - Professional Temporal Point Process Tool
=========================================================

Usage: easytpp.ps1 [COMMAND] [OPTIONS]

Commands:
  run              Run TPP experiments
  interactive      Interactive configuration mode
  list-configs     List available configuration files
  validate         Validate configuration file
  info             Show system information

Quick Examples:
  .\easytpp.ps1 --help                                    # Show detailed help
  .\easytpp.ps1 interactive                               # Interactive mode
  .\easytpp.ps1 run -c configs\config.yaml -e THP -d H2expc -p test
  .\easytpp.ps1 list-configs --dir configs
  .\easytpp.ps1 info

For detailed documentation, see CLI_README.md
"@ "Blue"
}

# Main execution
function Main {
    # Show header
    Write-ColorOutput @"
╔═══════════════════════════════════════════════════════════════╗
║                        EasyTPP CLI v2.0                      ║
║              Professional PowerShell Interface               ║
╚═══════════════════════════════════════════════════════════════╝
"@ "Blue"

    # Perform checks
    if (-not (Test-PythonAvailability)) {
        exit 1
    }
    
    if (-not (Test-CLIScript)) {
        exit 1
    }
    
    # Initialize directories
    Initialize-Directories
    
    # Handle arguments
    if ($Arguments.Count -eq 0) {
        Show-Help
        return
    }
    
    # Build command
    $command = "python `"$PythonScript`""
    foreach ($arg in $Arguments) {
        if ($arg -contains " ") {
            $command += " `"$arg`""
        } else {
            $command += " $arg"
        }
    }
    
    # Execute command
    try {
        Write-ColorOutput "🔧 Executing: $command" "Cyan"
        Write-ColorOutput "─" * 60 "Gray"
        
        Invoke-Expression $command
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "─" * 60 "Gray"
            Write-ColorOutput "✅ Command completed successfully" "Green"
        } else {
            Write-ColorOutput "─" * 60 "Gray"
            Write-ColorOutput "❌ Command failed with exit code: $LASTEXITCODE" "Red"
            exit $LASTEXITCODE
        }
    }
    catch {
        Write-ColorOutput "❌ Error executing command: $_" "Red"
        exit 1
    }
}

# Run main function
Main
