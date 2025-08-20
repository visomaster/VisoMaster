# Get the parent directory of the script location
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$VISO_ROOT = Split-Path -Parent $scriptPath

# Define dependencies directory
$DEPENDENCIES = Join-Path $VISO_ROOT "dependencies"

# Use external Python 3.11 installation
$EXTERNAL_PYTHON_PATH = "C:\bin\python\Python311"
$EXTERNAL_PYTHON_EXECUTABLE = Join-Path $EXTERNAL_PYTHON_PATH "python.exe"
$EXTERNAL_PYTHONW_EXECUTABLE = Join-Path $EXTERNAL_PYTHON_PATH "pythonw.exe"

# Create virtual environment if it doesn't exist
$VENV_PATH = Join-Path $VISO_ROOT "venv"
$VENV_SCRIPTS = Join-Path $VENV_PATH "Scripts"
$VENV_PYTHON = Join-Path $VENV_SCRIPTS "python.exe"
$VENV_PYTHONW = Join-Path $VENV_SCRIPTS "pythonw.exe"

if (-not (Test-Path $VENV_PATH)) {
    Write-Host "Creating virtual environment..."
    & $EXTERNAL_PYTHON_EXECUTABLE -m venv $VENV_PATH
    Write-Host "Virtual environment created at: $VENV_PATH"
}

# Define Python paths using the virtual environment
$PYTHON_PATH = $VENV_PATH
$PYTHON_SCRIPTS = $VENV_SCRIPTS
$PYTHON_EXECUTABLE = $VENV_PYTHON
$PYTHONW_EXECUTABLE = $VENV_PYTHONW

$GIT_EXECUTABLE = Join-Path $DEPENDENCIES "git-portable\bin\git.exe"

# Define CUDA and TensorRT paths
$CUDA_PATH = Join-Path $DEPENDENCIES "CUDA"
$CUDA_BIN_PATH = Join-Path $CUDA_PATH "bin"
$TENSORRT_PATH = Join-Path $DEPENDENCIES "TensorRt\lib"

# Define FFMPEG path correctly
$FFMPEG_PATH = $DEPENDENCIES

# Add all necessary paths to system PATH
$env:PATH = "$FFMPEG_PATH;$PYTHON_PATH;$PYTHON_SCRIPTS;$CUDA_BIN_PATH;$TENSORRT_PATH;$env:PATH"

# Set environment variables for the current session
$env:VISO_ROOT = $VISO_ROOT
$env:DEPENDENCIES = $DEPENDENCIES
$env:EXTERNAL_PYTHON_PATH = $EXTERNAL_PYTHON_PATH
$env:EXTERNAL_PYTHON_EXECUTABLE = $EXTERNAL_PYTHON_EXECUTABLE
$env:VENV_PATH = $VENV_PATH
$env:GIT_EXECUTABLE = $GIT_EXECUTABLE
$env:PYTHON_PATH = $PYTHON_PATH
$env:PYTHON_SCRIPTS = $PYTHON_SCRIPTS
$env:PYTHON_EXECUTABLE = $PYTHON_EXECUTABLE
$env:PYTHONW_EXECUTABLE = $PYTHONW_EXECUTABLE
$env:CUDA_PATH = $CUDA_PATH
$env:CUDA_BIN_PATH = $CUDA_BIN_PATH
$env:TENSORRT_PATH = $TENSORRT_PATH
$env:FFMPEG_PATH = $FFMPEG_PATH

Write-Host "Environment setup complete!"
Write-Host "Python: $PYTHON_EXECUTABLE"
Write-Host "Virtual Environment: $VENV_PATH"

