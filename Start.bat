@echo off
setlocal enabledelayedexpansion

:: Define project root dynamically
SET APP_ROOT=%~dp0
SET APP_ROOT=%APP_ROOT:~0,-1%

:: Activate virtual environment first
call "%APP_ROOT%\.venv\Scripts\activate.bat"

:: Then convert UI files after venv is activated
call app\ui\core\convert_ui_to_py.bat

:: Run the main program
python main.py

pause
