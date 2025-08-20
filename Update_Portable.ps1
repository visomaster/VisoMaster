# Check if install.dat exists
if (-not (Test-Path "install.dat")) {
    Write-Host "install.dat file not found!"
    Read-Host "Press Enter to continue..."
    exit 1
}

# Read the cuda_version from install.dat
$cudaVersion = Get-Content "install.dat" | Where-Object { $_ -match "cuda_version=" } | ForEach-Object { ($_ -split "=")[1] }

# Call the appropriate update script
& (Join-Path $PSScriptRoot "scripts\update_$cudaVersion.ps1")

Read-Host "Press Enter to continue..."
