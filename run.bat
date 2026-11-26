@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title TSNE Customer Clustering - WSL2 GPU/CPU Runner

:: [ADDED] Kiem tra GPU NVIDIA va tu dong cai WSL2 neu chua co
set "HAS_NVIDIA_GPU_AUTO=0"
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    set "HAS_NVIDIA_GPU_AUTO=1"
    echo [*] Phat hien GPU NVIDIA. Dang kiem tra trang thai WSL2...
    wsl.exe --status >nul 2>&1
    if errorlevel 1 (
        echo [*] WSL2 chua san sang. Dang tu dong kich hoat cai dat WSL2...
        wsl.exe --install --no-launch
        if errorlevel 1 (
            echo [!] Khong the tu dong cai WSL2. Hay chay file .bat nay bang quyen Administrator hoac chay: wsl --install
        ) else (
            echo [!] Da kich hoat cai dat WSL2. Vui long KHOI DONG LAI MAY roi chay lai file nay.
            pause
            exit /b 0
        )
    ) else (
        echo [OK] WSL2 da san sang.
    )
)

echo ============================================================
echo  TSNE CUSTOMER CLUSTERING - WSL2 RUNNER
echo  Repository: %CD%
echo ============================================================
echo.

where wsl.exe >nul 2>&1
if errorlevel 1 (
    echo [ERROR] WSL is not installed or wsl.exe is not on PATH.
    echo Install WSL2 and an Ubuntu distribution first:
    echo   wsl --install -d Ubuntu-24.04
    echo.
    pause
    exit /b 1
)

if not exist "%~dp0app.py" (
    echo [ERROR] app.py was not found beside run.bat.
    echo Place run.bat and run_wsl.sh in the repository root.
    echo.
    pause
    exit /b 1
)

if not exist "%~dp0run_wsl.sh" (
    echo [ERROR] Missing run_wsl.sh.
    echo Keep run.bat and run_wsl.sh together in the repository root.
    echo.
    pause
    exit /b 1
)

set "WSL_ROOT="
for /f "usebackq delims=" %%I in (`wsl.exe --exec wslpath -u "%CD%" 2^>nul`) do set "WSL_ROOT=%%I"

if not defined WSL_ROOT (
    echo [ERROR] Could not translate the repository path into WSL.
    echo Check that a WSL2 Linux distribution is installed and can start.
    echo Run "wsl --list --verbose" to inspect installed distributions.
    echo.
    pause
    exit /b 1
)

echo [OK] WSL repository path: %WSL_ROOT%
echo [*] Launching Streamlit inside WSL2.
echo [*] Waiting for the application, then opening the Windows browser.
echo [*] Linux browser auto-open is disabled to avoid the WSL gio warning.
echo.

rem Start WSL in a separate console so its logs remain visible.
if defined TSNE_WSL_DISTRO (
    start "TSNE Customer Clustering - WSL2" wsl.exe -d "%TSNE_WSL_DISTRO%" --cd "%WSL_ROOT%" --exec bash -lc "TSNE_STREAMLIT_PORT=8502 bash ./run_wsl.sh"
) else (
    start "TSNE Customer Clustering - WSL2" wsl.exe --cd "%WSL_ROOT%" --exec bash -lc "TSNE_STREAMLIT_PORT=8502 bash ./run_wsl.sh"
)

if errorlevel 1 (
    echo [ERROR] Could not start the WSL runner.
    echo.
    pause
    exit /b 1
)

echo [*] Waiting for Streamlit health endpoint at http://localhost:8502 ...
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$uri='http://127.0.0.1:8502/_stcore/health'; $deadline=(Get-Date).AddSeconds(120); $ready=$false; while ((Get-Date) -lt $deadline) { try { $r=Invoke-WebRequest -UseBasicParsing -Uri $uri -TimeoutSec 2; if ($r.StatusCode -eq 200) { $ready=$true; break } } catch { }; Start-Sleep -Seconds 1 }; if ($ready) { Start-Process 'http://localhost:8502'; exit 0 } else { Write-Host '[ERROR] Streamlit did not become ready within 120 seconds.'; exit 1 }"
if errorlevel 1 (
    echo.
    echo [ERROR] Streamlit did not respond at port 8502.
    echo Check the WSL runner window and logs folder for the actual error.
) else (
    echo [OK] Streamlit is ready. Windows browser opened: http://localhost:8502
)
echo.
echo The WSL Streamlit window can stay open while you use the application.
echo Press any key to close this launcher window.
pause >nul
exit /b 0