@echo off
REM Pipeline runner for Proximity Model
REM Runs ProximityModel.py, then test_maps.py on success

echo ================================================================================
echo Proximity Model Pipeline Runner
echo ================================================================================
echo.

REM Activate conda environment
call conda activate ParkximityENV

REM Check if activation was successful
if errorlevel 1 (
    echo ERROR: Failed to activate ParkximityENV conda environment
    echo Please ensure the environment exists and conda is properly configured
    pause
    exit /b 1
)

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0

REM Record pipeline start time
set PIPELINE_START=%time%

echo Running ProximityModel.py...
echo ================================================================================
echo.

set PHASE_START=%time%
python "%SCRIPT_DIR%Implementations\ProximityModel.py"

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
echo.
echo Proceeding to test maps...
echo.

set PHASE_START=%time%
python "%SCRIPT_DIR%Implementations\test_maps.py"

if errorlevel 1 (
    echo.
    echo ================================================================================
    echo ERROR: test_maps.py failed with error code %errorlevel%
    echo ================================================================================
    pause
    exit /b 1
)

call :elapsed %PHASE_START% %time%
echo.
echo ================================================================================
echo test_maps.py completed successfully  [%DURATION%]
echo ================================================================================

call :elapsed %PIPELINE_START% %time%
echo.
echo ================================================================================
echo Pipeline Complete!  Total runtime: %DURATION%
echo ================================================================================
echo.
echo Test maps are in: %SCRIPT_DIR%Output\test_maps\
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
