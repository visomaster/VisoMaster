# Source the environment setup script
. (Join-Path $PSScriptRoot "setenv.ps1")

Write-Host "Installing requirements into virtual environment..."

# Upgrade pip first
& $env:PYTHON_EXECUTABLE -m pip install --upgrade pip

# Install requirements based on CUDA version (default to cu128 for now)
$requirementsFile = "requirements_cu128.txt"
if (Test-Path $requirementsFile) {
    Write-Host "Installing requirements from: $requirementsFile"
    & $env:PYTHON_EXECUTABLE -m pip install -r $requirementsFile
} else {
    Write-Host "Requirements file not found: $requirementsFile"
    exit 1
}

Write-Host "Requirements installation complete!"
Write-Host "Virtual environment is ready at: $env:VENV_PATH"
