@echo on
cd /d %~dp1

set m=%1
echo %m%
set m_in=tmp_640x360.mp4
set m_cut_cm=%m:.mp4=.cut_cm.mp4%

if %m% == %m_in% (echo "Error:in and out are same" && exit /b 99)

set FFMPEG="..\_internal\ffmpeg\ffmpeg.exe"
set PYTHON="..\_internal\python-3.8.10\python.exe"

echo "Initializing working folder..."
if not exist temp mkdir temp
del temp\%m_in%
del temp\tmp_*.jpg

echo "Making mp4 640x360 resolve..."
rem %FFMPEG% -y -i %m% -s 640x360 -r 0.1 -t 950 temp\%m_in%
%FFMPEG% -y -i %m% -s 640x360 -r 0.1 temp\%m_in%

echo "Populating corners..."
%FFMPEG% -i temp\%m_in% -s 80x45 -vf crop=80:45:0:0             -r 0.1 -frames:v 1000 temp\tmptl_%%04d.jpg
%FFMPEG% -i temp\%m_in% -s 80x45 -vf crop=80:45:in_w-80:0       -r 0.1 -frames:v 1000 temp\tmptr_%%04d.jpg
%FFMPEG% -i temp\%m_in% -s 80x45 -vf crop=80:45:0:in_h-45       -r 0.1 -frames:v 1000 temp\tmpbl_%%04d.jpg
%FFMPEG% -i temp\%m_in% -s 80x45 -vf crop=80:45:in_w-80:in_h-45 -r 0.1 -frames:v 1000 temp\tmpbr_%%04d.jpg

echo "Concatinating corners..."
%PYTHON% ..\tf_keras_conv_cm_concat_corners.py temp --verbose

rem delete temporary files
del temp\tmptl_*.jpg
del temp\tmptr_*.jpg
del temp\tmpbl_*.jpg
del temp\tmpbr_*.jpg

echo "Preparing tsv for predict..."
%PYTHON% ..\tf_keras_conv_cm_prepare_data.py temp --verbose

REM -----------------------------------------------------------------------------------
REM echo "type any to CONTINUE. Note that train data have been prepared at the moment"
REM PAUSE
REM exit /b
REM -----------------------------------------------------------------------------------

echo "Predict..."
%PYTHON% ..\tf_keras_conv_cm_160x90x3_predict_only.py --data_tsv watermarks.tsv --verbose

call partition_mp4.bat %m%

call concat_mp4.bat

move tmp_concat.mp4 %m_cut_cm%

rem delete temporary files
echo "type any to delete temporary files. you can abort right here cause nothing more follows."
PAUSE
del temp\tmp_*.mp4
del temp\tmp_*.jpg
del partition_mp4.bat
del watermarks.tsv
del filelist.txt

echo "type any to DONE!"
PAUSE

exit /b
rem ----SUB--END---------------------------------------
