@echo off
call scripts\setenv.bat
"%GIT_EXECUTABLE%" fetch origin main
"%GIT_EXECUTABLE%" reset --hard origin/main
"%PYTHON_EXECUTABLE%" -m pip install -r requirements_cu128.txt --default-timeout 100
"%PYTHON_EXECUTABLE%" download_models.py
