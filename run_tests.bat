@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM run_tests.bat — run automated tests before every commit / deploy
REM Usage:  run_tests
REM         run_tests --cov
REM ─────────────────────────────────────────────────────────────────────────────

set PYTHON=C:\Users\DELL\AppData\Local\Programs\Python\Python310\python.exe

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   VoiceCount — Automated Tests
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

if "%1"=="--cov" (
    "%PYTHON%" -m pytest tests/ -v --tb=short --no-header -p no:warnings ^
        --cov=core --cov=features/utils --cov=features/manage_orders ^
        --cov=features/create_order --cov=features/seguimiento ^
        --cov-report=term-missing --cov-report=html:htmlcov
) else (
    "%PYTHON%" -m pytest tests/ -v --tb=short --no-header -p no:warnings
)

if %ERRORLEVEL% == 0 (
    echo.
    echo ✔  All tests passed. Safe to commit / deploy.
) else (
    echo.
    echo ✘  Tests FAILED. Do NOT commit until all tests pass.
    exit /b 1
)

echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
