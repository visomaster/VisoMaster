# Source the environment setup script
. (Join-Path $PSScriptRoot "setenv.ps1")

# Perform Git operations
& $env:GIT_EXECUTABLE fetch origin main
& $env:GIT_EXECUTABLE reset --hard origin/main

# Install requirements
& $env:PYTHON_EXECUTABLE -m pip install -r requirements_cu118.txt --default-timeout 100

# Download models
& $env:PYTHON_EXECUTABLE download_models.py

