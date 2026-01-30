@echo off
chcp 65001 >nul
REM AI Learning Website - Windows 一键部署脚本
REM 用法: scripts\deploy.bat "提交信息"

setlocal enabledelayedexpansion

echo ========================================
echo   AI Learning Website 部署脚本
echo ========================================
echo.

REM 切换到项目目录
cd /d "%~dp0\.."

REM 获取提交信息
set "COMMIT_MSG=%~1"
if "%COMMIT_MSG%"=="" set "COMMIT_MSG=update: 更新内容"

REM 检查git状态
git diff --quiet 2>nul
set DIFF_RESULT=%ERRORLEVEL%
git diff --cached --quiet 2>nul
set CACHED_RESULT=%ERRORLEVEL%

if %DIFF_RESULT%==0 if %CACHED_RESULT%==0 (
    echo [提示] 没有检测到更改，跳过提交
    goto :end
)

echo [1/4] 添加文件到暂存区...
git add .

echo [2/4] 创建提交...
git commit -m "%COMMIT_MSG%" -m "Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

echo [3/4] 推送到远程仓库...
git push origin main

echo [4/4] 完成!

:end
echo.
echo ========================================
echo 仓库: https://github.com/neko233-com/ai-learning-website
echo 网站: https://neko233-com.github.io/ai-learning-website/
echo ========================================

pause
