@echo off

:: Check if install.dat exists
if not exist install.dat (
    echo install.dat file not found!
    pause
    exit /b 1
)

:: Read the cuda_version from install.dat
for /f "tokens=2 delims==" %%A in ('findstr "cuda_version" install.dat') do set CUDA_VERSION=%%A

call scripts\update_%CUDA_VERSION%.bat
pause
