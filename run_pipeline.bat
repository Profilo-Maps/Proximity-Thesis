@echo off
REM Pipeline runner for Proximity Model
REM This script provides a convenient way to run the full pipeline

echo ================================================================================
echo Proximity Model Pipeline Runner
echo ================================================================================
echo.
echo This script provides two options:
echo.
echo   [1] Launch Configuration UI (recommended for first-time setup)
echo       - Configure cities and data paths through GUI
echo       - UI will launch analysis in a separate window
echo       - You can then run verification manually or use option 2
echo.
echo   [2] Run Analysis + Verification (for pre-configured setups)
echo       - Runs ProximityModel.py with existing configuration
echo       - Automatically runs verification after completion
echo.
echo   [3] Run Verification Only
echo       - Verifies existing parquet files
echo       - Creates diagnostic maps
echo.
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

:MENU
echo.
set /p CHOICE="Enter your choice (1, 2, or 3): "

if "%CHOICE%"=="1" goto LAUNCH_UI
if "%CHOICE%"=="2" goto RUN_ANALYSIS
if "%CHOICE%"=="3" goto RUN_VERIFICATION
echo Invalid choice. Please enter 1, 2, or 3.
goto MENU

:LAUNCH_UI
echo.
echo ================================================================================
echo Launching Configuration UI...
echo ================================================================================
echo.
echo The UI will open in a new window.
echo Configure your settings and click "Run Analysis" when ready.
echo The analysis will run in a separate console window.
echo.
python "%SCRIPT_DIR%ProximityModelUI.py"

if errorlevel 1 (
    echo.
    echo ERROR: UI closed with error code %errorlevel%
    pause
    exit /b 1
)

echo.
echo UI closed.
echo.
echo If you started an analysis, it should be running in a separate window.
echo You can run verification (option 3) after the analysis completes.
echo.
pause
exit /b 0

:RUN_ANALYSIS
echo.
echo ================================================================================
echo Running ProximityModel.py...
echo ================================================================================
echo.
python "%SCRIPT_DIR%ProximityModel.py"

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
echo Proceeding to verification...
echo.
goto RUN_VERIFICATION

:RUN_VERIFICATION
echo.
echo ================================================================================
echo Running verify_parquet.py...
echo ================================================================================
echo.
python "%SCRIPT_DIR%verify_parquet.py"

if errorlevel 1 (
    echo.
    echo ERROR: verify_parquet.py failed with error code %errorlevel%
    echo Verification failed. Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Verification Complete!
echo ================================================================================
echo.
echo Output files are in: %SCRIPT_DIR%Output\
echo Diagnostic maps are in: %SCRIPT_DIR%Output\Diagnostic Maps\
echo.
pause
exit /b 0
