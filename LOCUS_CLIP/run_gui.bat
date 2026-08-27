@echo off
REM Launch Clipping Studio (no console window).
REM The venv lives at the repo root, one level up from this LOCUS_CLIP folder.
start "" "%~dp0..\.venv\Scripts\pythonw.exe" "%~dp0gui.py"
