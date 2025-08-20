# Source the environment setup script to get Python paths
. (Join-Path $PSScriptRoot "scripts\setenv.ps1")

# Convert UI files
& (Join-Path $PSScriptRoot "app\ui\core\convert_ui_to_py.ps1")

# Set environment variables
$APP_ROOT = $PSScriptRoot
$DEPENDENCIES = Join-Path $APP_ROOT "dependencies"
Write-Host $DEPENDENCIES

# Add dependencies to PATH
$env:PATH = "$DEPENDENCIES;$env:PATH"

# Run the main application using the bundled Python
& $env:PYTHON_EXECUTABLE main.py

Read-Host "Press Enter to continue..."
