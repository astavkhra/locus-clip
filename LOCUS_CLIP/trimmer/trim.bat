@echo off
REM Convenience wrapper: runs trim.py with the project venv's Python.
"%~dp0..\.venv\Scripts\python.exe" "%~dp0trim.py" %*
