@echo on

cd %~dp0

set FFMPEG="ffmpeg.exe"


%FFMPEG% -i 20180719.mp4 -s 80x45 -vf crop=80:45:0:0             -r 0.1 -frames:v 1000 tmptl_%%04d.jpg
%FFMPEG% -i 20180719.mp4 -s 80x45 -vf crop=80:45:in_w-80:0       -r 0.1 -frames:v 1000 tmptr_%%04d.jpg
%FFMPEG% -i 20180719.mp4 -s 80x45 -vf crop=80:45:0:in_h-45       -r 0.1 -frames:v 1000 tmpbl_%%04d.jpg
%FFMPEG% -i 20180719.mp4 -s 80x45 -vf crop=80:45:in_w-80:in_h-45 -r 0.1 -frames:v 1000 tmpbr_%%04d.jpg




PAUSE
