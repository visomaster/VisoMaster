@echo off
setlocal enabledelayedexpansion

:: Define relative paths
set "UI_FILE=app\ui\core\MainWindow.ui"
set "PY_FILE=app\ui\core\main_window.py"
set "QRC_FILE=app\ui\core\media.qrc"
set "RCC_PY_FILE=app\ui\core\media_rc.py"

:: Print paths for debugging
echo Checking UI file: %UI_FILE%
echo Checking QRC file: %QRC_FILE%

:: Check if files exist before running commands
if not exist "%UI_FILE%" (
    echo ERROR: UI file not found: %UI_FILE%
    exit /b 1
)
if not exist "%QRC_FILE%" (
    echo ERROR: QRC file not found: %QRC_FILE%
    exit /b 1
)

:: Run PySide6 commands
pyside6-uic "%UI_FILE%" -o "%PY_FILE%"
pyside6-rcc "%QRC_FILE%" -o "%RCC_PY_FILE%"

:: Verify output files exist
if not exist "%PY_FILE%" (
    echo ERROR: UI conversion failed, file not generated: %PY_FILE%
    exit /b 1
)
if not exist "%RCC_PY_FILE%" (
    echo ERROR: RCC conversion failed, file not generated: %RCC_PY_FILE%
    exit /b 1
)

:: Define search and replace strings
set "searchString=import media_rc"
set "replaceString=from app.ui.core import media_rc"

:: Create a temporary file
set "tempFile=%PY_FILE%.tmp"

:: Process the file
(for /f "usebackq delims=" %%A in ("%PY_FILE%") do (
    set "line=%%A"
    if "!line!"=="%searchString%" (
        echo %replaceString%
    ) else (
        echo !line!
    )
)) > "%tempFile%"

:: Replace the original file with the temporary file
move /y "%tempFile%" "%PY_FILE%"

echo Replacement complete.
