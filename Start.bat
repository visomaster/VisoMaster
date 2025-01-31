
call conda activate visomaster
call app/ui/core/convert_ui_to_py.bat
SET APP_ROOT=%~dp0
SET APP_ROOT=%APP_ROOT:~0,-1%
SET DEPENDENCIES=%APP_ROOT%\dependencies
echo %DEPENDENCIES%
SET PATH=%DEPENDENCIES%;%PATH%
python main.py
pause