@echo off

echo Project file location: %~dp0

echo Running Python Pack Initialization
color 0E
timeout /t 2 /nobreak >nul

if not exist ".\generated_environment" (
    echo Generating Virtual Environment
    color 07
    color 0C
    echo [DO NOT CLOSE]

    py -m venv %~dp0generated_environment
) else (
    color 0E
    echo Environment already exists
)
color 07

color 0E
echo ACTIVATING ENVIRONMENT 
color 0C
echo [DO NOT CLOSE]
color 07

call generated_environment\Scripts\activate.bat

color 0A
echo INSTALLING PIP MODULES

:: Installed Pip Modules
py -m pip install customtkinter
py -m pip install nltk

color 07
echo Installed pip modules

echo Creating Log
if not exist ".\log.txt" (
    echo. > ".\log.txt"
    echo Created Log File
) else (
    color 0E
    echo Log File already exists
)

color 0E
echo Setup Complete!!
pause