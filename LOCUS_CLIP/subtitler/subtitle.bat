@echo off
REM Convenience wrapper: runs subtitle.py with the project venv's Python.
"%~dp0..\.venv\Scripts\python.exe" "%~dp0subtitle.py" %*
