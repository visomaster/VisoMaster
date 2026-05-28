@echo off
setlocal EnableDelayedExpansion
title VisoMaster

SET "APP_ROOT=%~dp0"
IF "%APP_ROOT:~-1%"=="\" SET "APP_ROOT=%APP_ROOT:~0,-1%"
cd /d "%APP_ROOT%"

:: ── Detect Python: portable bundled Python takes priority, fall back to conda ─
SET "PYTHON_EXECUTABLE="
IF EXIST "%APP_ROOT%\dependencies\Python\python.exe" (
    CALL "%APP_ROOT%\scripts\setenv.bat"
) ELSE (
    CALL conda activate visomaster 2>nul
    IF ERRORLEVEL 1 (
        echo [ERROR] No bundled Python found and conda environment "visomaster" could not be activated.
        echo         Run the installer or set up the conda environment first.
        pause & exit /b 1
    )
    SET "PYTHON_EXECUTABLE=python"
)

:: ── Load .env if present ──────────────────────────────────────────────────────
IF EXIST "%APP_ROOT%\.env" (
    FOR /F "usebackq tokens=1,* delims==" %%A IN ("%APP_ROOT%\.env") DO (
        SET "LINE=%%A"
        IF NOT "!LINE:~0,1!"=="#" IF NOT "%%A"=="" SET "%%A=%%B"
    )
)

:: ── Add bundled dependencies to PATH (FFmpeg etc.) ───────────────────────────
IF EXIST "%APP_ROOT%\dependencies" SET "PATH=%APP_ROOT%\dependencies;%PATH%"

:: ── Detect bun or npm once ────────────────────────────────────────────────────
SET "BUN_CMD="
where bun >nul 2>&1 && SET "BUN_CMD=bun"
IF "%BUN_CMD%"=="" (
    where npm >nul 2>&1 && SET "BUN_CMD=npm"
)

:menu
cls
echo.
echo  ==============================================
echo    VisoMaster
echo  ==============================================
echo.
echo    1. Qt Desktop         (native Qt UI)
echo    2. WebView            (Qt + embedded web UI)
echo    3. Web                (API server + browser UI)
echo.
echo    Q. Quit
echo.
SET /P "CHOICE=  Select mode: "

IF /I "%CHOICE%"=="1" GOTO :mode_qt
IF /I "%CHOICE%"=="2" GOTO :mode_webview
IF /I "%CHOICE%"=="3" GOTO :mode_web
IF /I "%CHOICE%"=="Q" GOTO :quit
IF /I "%CHOICE%"=="quit" GOTO :quit

echo.
echo  Invalid choice. Try again.
timeout /t 1 /nobreak >nul
GOTO :menu

:: ── Mode 1: Native Qt UI ──────────────────────────────────────────────────────
:mode_qt
cls
echo.
echo  Starting Qt Desktop UI...
echo.
"%PYTHON_EXECUTABLE%" main.py
GOTO :done

:: ── Mode 2: Qt + embedded web UI ─────────────────────────────────────────────
:mode_webview
cls
echo.
echo  Starting WebView UI...
echo.

IF "%BUN_CMD%"=="" (
    echo  [ERROR] bun or npm not found. Install bun from https://bun.sh
    pause & exit /b 1
)

IF NOT EXIST "%APP_ROOT%\logs" mkdir "%APP_ROOT%\logs"

:: Start Vite dev server hidden, save PID to file
echo  [1/2] Starting Vite dev server ^(%BUN_CMD% run dev^)...
powershell -NoProfile -Command "$p = Start-Process -FilePath '%BUN_CMD%' -ArgumentList 'run','dev' -WorkingDirectory '%APP_ROOT%\visomaster-ui' -WindowStyle Hidden -RedirectStandardOutput '%APP_ROOT%\logs\vite.log' -RedirectStandardError '%APP_ROOT%\logs\vite.err.log' -PassThru; $p.Id | Out-File -FilePath '%APP_ROOT%\logs\vite.pid' -Encoding ascii"
SET /P VITE_PID=<"%APP_ROOT%\logs\vite.pid"

:: Give Vite a moment to bind before Qt opens the webview
echo  Waiting for Vite to start...
timeout /t 4 /nobreak >nul

:: Start API server hidden, save PID to file
echo  [2/2] Starting FastAPI server on http://localhost:8000 ...
powershell -NoProfile -Command "$p = Start-Process -FilePath '%PYTHON_EXECUTABLE%' -ArgumentList '-m','app.api.server' -WorkingDirectory '%APP_ROOT%' -WindowStyle Hidden -RedirectStandardOutput '%APP_ROOT%\logs\api.log' -RedirectStandardError '%APP_ROOT%\logs\api.err.log' -PassThru; $p.Id | Out-File -FilePath '%APP_ROOT%\logs\api.pid' -Encoding ascii"
SET /P API_PID=<"%APP_ROOT%\logs\api.pid"

:: Give the API server a moment before the Qt window tries to connect
timeout /t 2 /nobreak >nul

echo  Launching Qt WebView window...
echo.
"%PYTHON_EXECUTABLE%" web_main.py

:: Main process exited — kill background servers
echo.
echo  Shutting down background servers...
IF DEFINED VITE_PID (
    powershell -NoProfile -Command "Stop-Process -Id %VITE_PID% -Force -ErrorAction SilentlyContinue"
)
IF DEFINED API_PID (
    powershell -NoProfile -Command "Stop-Process -Id %API_PID% -Force -ErrorAction SilentlyContinue"
)
GOTO :done

:: ── Mode 3: Web-only (API server + React frontend) ───────────────────────────
:mode_web
cls
echo.
echo  Starting Web mode...
echo.

IF "%BUN_CMD%"=="" (
    echo  [ERROR] bun or npm not found. Install bun from https://bun.sh
    pause & exit /b 1
)

IF NOT EXIST "%APP_ROOT%\logs" mkdir "%APP_ROOT%\logs"

:: Start API server hidden, save PID to file
echo  [1/2] Starting FastAPI server on http://localhost:8000 ...
powershell -NoProfile -Command "$p = Start-Process -FilePath '%PYTHON_EXECUTABLE%' -ArgumentList '-m','app.api.server' -WorkingDirectory '%APP_ROOT%' -WindowStyle Hidden -RedirectStandardOutput '%APP_ROOT%\logs\api.log' -RedirectStandardError '%APP_ROOT%\logs\api.err.log' -PassThru; $p.Id | Out-File -FilePath '%APP_ROOT%\logs\api.pid' -Encoding ascii"
SET /P API_PID=<"%APP_ROOT%\logs\api.pid"

echo  Waiting for API server to start...
timeout /t 3 /nobreak >nul

:: Start Vite dev server hidden, save PID to file
echo  [2/2] Starting Vite dev server ^(%BUN_CMD% run dev^)...
powershell -NoProfile -Command "$p = Start-Process -FilePath '%BUN_CMD%' -ArgumentList 'run','dev' -WorkingDirectory '%APP_ROOT%\visomaster-ui' -WindowStyle Hidden -RedirectStandardOutput '%APP_ROOT%\logs\vite.log' -RedirectStandardError '%APP_ROOT%\logs\vite.err.log' -PassThru; $p.Id | Out-File -FilePath '%APP_ROOT%\logs\vite.pid' -Encoding ascii"
SET /P VITE_PID=<"%APP_ROOT%\logs\vite.pid"

echo.
echo  Both servers are running in the background.
echo.
echo    API server :  http://localhost:8000
echo    Web UI     :  http://localhost:5173
echo.
echo  Press any key to stop both servers and exit.
pause >nul

:: Kill background servers on exit
echo  Shutting down background servers...
IF DEFINED VITE_PID (
    powershell -NoProfile -Command "Stop-Process -Id %VITE_PID% -Force -ErrorAction SilentlyContinue"
)
IF DEFINED API_PID (
    powershell -NoProfile -Command "Stop-Process -Id %API_PID% -Force -ErrorAction SilentlyContinue"
)
GOTO :done

:quit
exit /b 0

:done
echo.
pause
endlocal
