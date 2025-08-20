# Source the environment setup script
. (Join-Path $PSScriptRoot "scripts\setenv.ps1")

# Run the main application
& $env:PYTHON_EXECUTABLE main.py

# Keep the window open
Read-Host "Press Enter to continue..."

