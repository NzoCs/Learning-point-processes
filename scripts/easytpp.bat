@echo off
REM EasyTPP CLI Batch Wrapper for Windows
REM Simple batch file to run the EasyTPP CLI tool

setlocal enabledelayedexpansion

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
set "PYTHON_SCRIPT=%SCRIPT_DIR%easytpp_cli.py"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python not found in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

REM Check if CLI script exists
if not exist "%PYTHON_SCRIPT%" (
    echo ❌ Error: EasyTPP CLI script not found: %PYTHON_SCRIPT%
    pause
    exit /b 1
)

REM Create directories if they don't exist
if not exist "%SCRIPT_DIR%configs" mkdir "%SCRIPT_DIR%configs"
if not exist "%SCRIPT_DIR%outputs" mkdir "%SCRIPT_DIR%outputs"
if not exist "%SCRIPT_DIR%logs" mkdir "%SCRIPT_DIR%logs"

REM Show header
echo ╔═══════════════════════════════════════════════════════════════╗
echo ║                        EasyTPP CLI v2.0                      ║
echo ║                Professional Batch Interface                  ║
echo ╚═══════════════════════════════════════════════════════════════╝

REM Execute the Python script with all arguments
python "%PYTHON_SCRIPT%" %*

REM Check exit code
if errorlevel 1 (
    echo.
    echo ❌ Command failed with exit code: %errorlevel%
    pause
    exit /b %errorlevel%
) else (
    echo.
    echo ✅ Command completed successfully
)

endlocal
