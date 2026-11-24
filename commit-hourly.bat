@echo off
@cd /d "%~dp0"
@for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
@set "DATETIME=%dt:~0,4%-%dt:~4,2%-%dt:~6,2%_%dt:~8,2%-%dt:~10,2%-%dt:~12,2%"
git add -A
git commit -m "Auto-commit: %DATETIME%"

