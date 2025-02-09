@echo off
echo Running Python Pack Initialization

echo installing Git Submodules

call git submodule init
call git submodule update
if errorlevel 1 {
    echo Failed to Initialize or Update Submodule.
    exit /b 1
}

echo Creating Log
if not exist ".\log.txt" {
    echo. > ".\log.txt"
    echo Created Log File
} else {
    echo Log File already exists
}

echo Setup Complete!!
pause