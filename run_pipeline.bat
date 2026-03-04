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

echo Running ProximityModel.py...
echo ================================================================================
echo.
python "%SCRIPT_DIR%Implementations\ProximityModel.py"

if errorlevel 1 (
    echo.
    echo ================================================================================
    echo ERROR: ProximityModel.py failed with error code %errorlevel%
    echo ================================================================================
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo ProximityModel.py completed successfully
echo ================================================================================
echo.
echo Proceeding to test maps...
echo.

python "%SCRIPT_DIR%Implementations\test_maps.py"

if errorlevel 1 (
    echo.
    echo ================================================================================
    echo ERROR: test_maps.py failed with error code %errorlevel%
    echo ================================================================================
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Pipeline Complete!
echo ================================================================================
echo.
echo Test maps are in: %SCRIPT_DIR%Output\test_maps\
echo.
pause
exit /b 0
