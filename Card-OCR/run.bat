@echo off
REM Bank Card OCR System - 快速启动脚本
REM 
REM 使用方法: 双击此文件即可启动程序

echo ========================================
echo Bank Card OCR System
echo 银行卡识别系统
echo ========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.7+
    echo.
    pause
    exit /b 1
)

echo [信息] 正在启动程序...
echo.

REM 启动主程序
python main.py

REM 如果程序异常退出，暂停以查看错误信息
if errorlevel 1 (
    echo.
    echo [错误] 程序异常退出
    pause
)
