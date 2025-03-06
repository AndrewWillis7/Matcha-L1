@echo off

:: Display the project file location
echo Project file location: %~dp0

:: Function to check if Python 3.11 is installed
:CheckPython311
echo Checking for Python 3.11...

:: Attempt to check Python version
py -3.11 --version >nul 2>&1

:: If Python 3.11 is not found, show error and pause
if %ERRORLEVEL% neq 0 (
    color 0C
    echo ERROR: Python 3.11 is not installed. Please install Python 3.11 to proceed.
    pause
    exit /b
)

:: If Python 3.11 is found, continue with the script
color 0A
echo Python 3.11 found. Continuing setup...

:: Begin Python pack initialization
echo Running Python Pack Initialization
color 0E
timeout /t 2 /nobreak >nul

:: Check if the virtual environment already exists
if not exist ".\generated_environment" (
    echo Generating Virtual Environment for Python 3.11
    color 07
    color 0C
    echo [DO NOT CLOSE]

    :: Create virtual environment with Python 3.11
    py -3.11 -m venv %~dp0generated_environment
) else (
    color 0E
    echo Environment already exists
)
color 07

:: Activate the virtual environment
color 0E
echo ACTIVATING ENVIRONMENT 
color 0C
echo [DO NOT CLOSE]
color 07

:: Upgrade pip to the latest version
py -m pip install --upgrade pip

call generated_environment\Scripts\activate.bat

:: Indicate that modules will be installed
color 0A
echo INSTALLING PIP MODULES

::PYTORCH FIX
py -m pip uninstall torch

:: Install necessary pip modules for the project
py -m pip install customtkinter
py -m pip install transformers
py -m pip install uvicorn
py -m pip install --upgrade transformers
py -m pip install torch-directml==0.2.5.dev240914
::py -m pip install torch==2.3.1

color 07
echo Installed pip modules

:: Create a log file if it doesn't already exist
echo Creating Log
if not exist ".\log.txt" (
    echo. > ".\log.txt"
    echo Created Log File
) else (
    color 0E
    echo Log File already exists
)

:: Install the model if it hasn't been installed yet
if not exist ".\lib\.locks" (
    echo INSTALLING MODEL [THIS WILL TAKE A WHILE!!]
    py model_install.py
) else (
    echo MODEL ALREADY INSTALLED, UPDATING
)

:: Notify that setup is complete
echo You Are Good to Close!

python -c "import torch; print(torch.__version__)"

:: Directing output of batch file execution to log.txt
color 0E
build.bat > log.txt

:: Indicate completion of setup
echo You Are Good to Close!
echo Setup Complete!!

:: Pause to keep the window open until a key is pressed
pause