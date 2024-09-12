call conda activate pyside_test
pyside6-uic MainWindow.ui -o MainWindow.py
pyside6-rcc media.qrc -o media_rc.py

@echo off
setlocal enabledelayedexpansion

:: Define file paths and strings
set "filePath=MainWindow.py"
set "searchString=import media_rc"
set "replaceString=from App.UI.Core import media_rc"

:: Create a temporary file
set "tempFile=%filePath%.tmp"

:: Process the file
(for /f "usebackq delims=" %%A in ("%filePath%") do (
    set "line=%%A"
    if "!line!"=="%searchString%" (
        echo %replaceString%
    ) else (
        echo !line!
    )
)) > "%tempFile%"

:: Replace the original file with the temporary file
move /y "%tempFile%" "%filePath%"

echo Replacement complete.