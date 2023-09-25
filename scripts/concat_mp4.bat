@echo off

set FFMPEG="_internal\ffmpeg\ffmpeg.exe"
set PYTHON="_internal\python-3.8.10\python.exe"

%FFMPEG% -f concat -i filelist.txt -c copy tmp_concat.mp4

rem PAUSE