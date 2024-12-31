call conda activate rope_pyside
pyside6-uic C:\Users\argen\Documents\Rope-PySide\app\ui\core\MainWindow.ui -o C:\Users\argen\Documents\Rope-PySide\app\ui\core\main_window.py
pyside6-rcc C:\Users\argen\Documents\Rope-PySide\app\ui\core\media.qrc -o C:\Users\argen\Documents\Rope-PySide\app\ui\core\media_rc.py

@echo off
setlocal enabledelayedexpansion

:: Define file paths and strings
set "filePath=C:\Users\argen\Documents\Rope-PySide\app\ui\core\main_window.py"
set "searchString=import media_rc"
set "replaceString=from app.ui.core import media_rc"

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