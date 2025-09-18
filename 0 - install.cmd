@echo off
REM =====================================================================
REM
REM           Installation Script for AlphaZero Gomoku Project
REM
REM - This script will create a Python virtual environment.
REM - It will then install all necessary dependencies from requirements.txt.
REM
REM =====================================================================

echo.
echo ====================================================
echo      AlphaZero Gomoku Environment Installer
echo ====================================================
echo.

REM --- 检查 Python 是否安装 ---
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in your system's PATH.
    echo Please install Python 3.8+ and ensure it is added to your PATH.
    goto :end
)
echo [INFO] Found Python installation.

REM --- 创建虚拟环境 ---
if not exist .venv (
    echo [INFO] Creating Python virtual environment in '.\venv\'...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create the virtual environment.
        goto :end
    )
    echo [SUCCESS] Virtual environment created successfully.
) else (
    echo [INFO] Virtual environment '.\venv\' already exists. Skipping creation.
)

REM --- 激活虚拟环境并安装依赖 ---
echo.
echo [INFO] Activating virtual environment and installing dependencies...
call .\.venv\Scripts\activate.bat

REM 检查 requirements.txt 文件是否存在
if not exist requirements.txt (
    echo [ERROR] 'requirements.txt' not found in the current directory.
    goto :end
)

echo [INFO] Installing packages from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install packages. Please check the error messages above.
    echo [TIP] You might need to install PyTorch manually for your specific CUDA version.
    echo Visit: https://pytorch.org/get-started/locally/
    goto :end
)

echo.
echo [SUCCESS] All dependencies have been installed successfully!
echo.
echo ===============================================================
echo      You can now run 'train.cmd' to start the training.
echo ===============================================================
echo.

:end
pause