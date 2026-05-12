@echo off
REM Pipeline runner for Proximity Model
REM One-stop launcher: pipeline, editor, or both.

echo ================================================================================
echo Proximity Model Pipeline Runner
echo ================================================================================
echo.
echo   [1] Full         - Pipeline + Editor
echo   [2] Pipeline only - Run pipeline
echo   [3] Editor only   - Launch editor
echo.
choice /c 123 /n /m "Select mode [1/2/3]: "
set MODE=%errorlevel%

REM Get the directory where this batch file is located and cd into it
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Record start time
set PIPELINE_START=%time%

REM --- Pipeline phase ---
if %MODE%==3 goto :editor

echo.
choice /c YN /n /m "Recreate existing parquets? [Y/N]: "
if errorlevel 2 (set SKIP_EXISTING=--skip-existing) else (set SKIP_EXISTING=)

echo.
echo Running ProximityModel.py...
echo ================================================================================
echo.

set PHASE_START=%time%
uv run python "%SCRIPT_DIR%Implementations\ProximityModel.py" %SKIP_EXISTING%

if errorlevel 1 (
    echo.
    echo ================================================================================
    echo ERROR: ProximityModel.py failed with error code %errorlevel%
    echo ================================================================================
    pause
    exit /b 1
)

call :elapsed %PHASE_START% %time%
echo.
echo ================================================================================
echo ProximityModel.py completed successfully  [%DURATION%]
echo ================================================================================

if %MODE%==2 goto :done

REM --- Editor phase ---
:editor

REM Skip launch if the API server is already running on port 8000
netstat -ano | findstr ":8000 " | findstr LISTENING >nul 2>&1
if not errorlevel 1 (
    echo.
    echo Editor is already running on port 8000 -- skipping re-launch.
    echo Close the existing editor windows first if you want to restart.
    echo.
    goto :done
)

echo.
echo Launching county editor...
echo ================================================================================
echo.

REM Start FastAPI backend
start "Proximity Editor API" cmd /k "cd /d "%SCRIPT_DIR%editor\server" && uv run uvicorn editor_server:app --reload --port 8000"

REM Start Vite dev server
start "Proximity Editor Web" cmd /k "cd /d "%SCRIPT_DIR%editor\web" && npm run dev"

REM Give the servers a moment to start before opening the browser
timeout /t 3 /nobreak >nul
start "" "http://localhost:5173"

echo Editor is running:
echo   API server:  http://localhost:8000
echo   Web app:     http://localhost:5173
echo.
echo Close the server windows to stop.
echo.

:done

call :elapsed %PIPELINE_START% %time%
echo.
echo ================================================================================
echo Done!  Total runtime: %DURATION%
echo ================================================================================
echo.
pause
exit /b 0

REM --- Subroutine: compute elapsed time ---
:elapsed
set START=%~1
set END=%~2

REM Parse start time
for /f "tokens=1-4 delims=:." %%a in ("%START%") do (
    set /a "S_H=%%a, S_M=1%%b-100, S_S=1%%c-100, S_CS=1%%d-100"
)
REM Parse end time
for /f "tokens=1-4 delims=:." %%a in ("%END%") do (
    set /a "E_H=%%a, E_M=1%%b-100, E_S=1%%c-100, E_CS=1%%d-100"
)

set /a "S_TOTAL=(S_H*360000)+(S_M*6000)+(S_S*100)+S_CS"
set /a "E_TOTAL=(E_H*360000)+(E_M*6000)+(E_S*100)+E_CS"
set /a "DIFF=E_TOTAL-S_TOTAL"

REM Handle midnight rollover
if %DIFF% lss 0 set /a "DIFF+=8640000"

set /a "D_H=DIFF/360000, DIFF%%=360000"
set /a "D_M=DIFF/6000, DIFF%%=6000"
set /a "D_S=DIFF/100, D_CS=DIFF%%100"

if %D_H% lss 10 set D_H=0%D_H%
if %D_M% lss 10 set D_M=0%D_M%
if %D_S% lss 10 set D_S=0%D_S%
if %D_CS% lss 10 set D_CS=0%D_CS%

set DURATION=%D_H%:%D_M%:%D_S%.%D_CS%
goto :eof
