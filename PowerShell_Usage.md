# PowerShell Scripts for VisoMaster

This directory now contains PowerShell (`.ps1`) versions of all the batch files for easier integration with modern Windows environments.

## Available PowerShell Scripts

### Core Scripts
- **`scripts/setenv.ps1`** - Sets up environment variables and PATH (equivalent to `setenv.bat`)
- **`Start_Portable.ps1`** - Launches VisoMaster in portable mode using bundled Python (equivalent to `Start_Portable.bat`)
- **`Start.ps1`** - Launches VisoMaster with bundled Python environment and UI conversion (equivalent to `Start.bat`)
- **`Update_Portable.ps1`** - Updates VisoMaster based on CUDA version (equivalent to `Update_Portable.bat`)

### Update Scripts
- **`scripts/update_cu118.ps1`** - Updates for CUDA 11.8 (equivalent to `update_cu118.bat`)
- **`scripts/update_cu124.ps1`** - Updates for CUDA 12.4 (equivalent to `update_cu124.bat`)
- **`scripts/update_cu128.ps1`** - Updates for CUDA 12.8 (equivalent to `update_cu128.bat`)

### Utility Scripts
- **`app/ui/core/convert_ui_to_py.ps1`** - Converts UI files (equivalent to `convert_ui_to_py.bat`)
- **`scripts/install_requirements.ps1`** - Installs requirements into the virtual environment

## How to Use

### Initial Setup (First Time Only)
```powershell
# Navigate to the VisoMaster directory
cd C:\path\to\VisoMaster

# Source the environment setup (creates virtual environment)
. .\scripts\setenv.ps1

# Install requirements into the virtual environment
.\scripts\install_requirements.ps1
```

### Method 1: Source the script (Recommended)
```powershell
# Navigate to the VisoMaster directory
cd C:\path\to\VisoMaster

# Source the environment setup
. .\scripts\setenv.ps1

# Now you can use the environment variables
$env:PYTHON_EXECUTABLE --version

# Or run Python directly
& $env:PYTHON_EXECUTABLE --version
```

### Method 2: Run scripts directly
```powershell
# Launch VisoMaster portable
.\Start_Portable.ps1

# Update for CUDA 12.4
.\scripts\update_cu124.ps1
```

### Method 3: Execute in current session
```powershell
# Execute and keep variables in current session
& .\scripts\setenv.ps1
```

## Key Differences from Batch Files

1. **Environment Variables**: PowerShell scripts set environment variables for the current session using `$env:VARIABLE_NAME`
2. **Path Handling**: Uses `Join-Path` for cross-platform compatible path construction
3. **Script Sourcing**: Can be sourced into the current session using `. .\script.ps1`
4. **Error Handling**: Better error handling and PowerShell-native syntax
5. **Session Persistence**: Environment variables persist in the current PowerShell session
6. **Python Environment**: Uses bundled Python from dependencies folder instead of conda

## Environment Variables Set

The `setenv.ps1` script sets the following environment variables:
- `VISO_ROOT` - Root directory of the project
- `DEPENDENCIES` - Path to dependencies folder
- `GIT_EXECUTABLE` - Path to portable Git
- `PYTHON_PATH`, `PYTHON_SCRIPTS`, `PYTHON_EXECUTABLE`, `PYTHONW_EXECUTABLE` - Python paths
- `CUDA_PATH`, `CUDA_BIN_PATH` - CUDA paths
- `TENSORRT_PATH` - TensorRT library path
- `FFMPEG_PATH` - FFMPEG path
- `PATH` - Updated system PATH

## Virtual Environment Setup

The PowerShell scripts now use your external Python 3.11 installation at `C:\bin\python\Python311` and create a virtual environment in the project root. This provides:

- **Modern Python**: Full support for PyTorch 2.8.0+ and modern typing features
- **Isolated Environment**: Clean, isolated package installation
- **Flexible**: Can use different Python versions as needed
- **Compatible**: Supports all the latest PyTorch and CUDA requirements

## Troubleshooting

### Execution Policy Issues
If you encounter execution policy restrictions, you may need to change the execution policy:
```powershell
# Check current policy
Get-ExecutionPolicy

# Set policy for current user (if needed)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Path Issues
If paths are not found, ensure you're running the scripts from the correct directory (VisoMaster root).

## Compatibility

- **Windows**: Full compatibility with Windows 10/11
- **PowerShell**: Requires PowerShell 5.1 or later (Windows 10+)
- **Dependencies**: Same as batch file versions
