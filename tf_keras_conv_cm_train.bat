@echo on
cd /d %~dp1


set PYTHON="_internal\python-3.8.10\python.exe"
REM set PATH=_internal\CUDA;%PATH%

echo "Predict..."
%PYTHON% tf_keras_conv_cm_160x90x3.py

echo "type any to DONE!"
PAUSE

exit /b
