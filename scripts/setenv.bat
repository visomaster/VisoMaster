@echo off

:: Get the parent directory of the script location
SET "VISO_ROOT=%~dp0"
SET "VISO_ROOT=%VISO_ROOT:~0,-1%"
FOR %%A IN ("%VISO_ROOT%\..") DO SET "VISO_ROOT=%%~fA"

:: Define dependencies directory
SET "DEPENDENCIES=%VISO_ROOT%\dependencies"

SET "GIT_EXECUTABLE=%DEPENDENCIES%\git-portable\bin\git.exe"

:: Define Python paths
SET "PYTHON_PATH=%DEPENDENCIES%\Python"
SET "PYTHON_SCRIPTS=%PYTHON_PATH%\Scripts"
SET "PYTHON_EXECUTABLE=%PYTHON_PATH%\python.exe"
SET "PYTHONW_EXECUTABLE=%PYTHON_PATH%\pythonw.exe"

:: Define CUDA and TensorRT paths
SET "CUDA_PATH=%DEPENDENCIES%\CUDA"
SET "CUDA_BIN_PATH=%CUDA_PATH%\bin"
SET "TENSORRT_PATH=%DEPENDENCIES%\TensorRt\lib"

:: Define FFMPEG path correctly
SET "FFMPEG_PATH=%DEPENDENCIES%"

:: Add all necessary paths to system PATH
SET "PATH=%FFMPEG_PATH%;%PYTHON_PATH%;%PYTHON_SCRIPTS%;%CUDA_BIN_PATH%;%TENSORRT_PATH%;%PATH%"
