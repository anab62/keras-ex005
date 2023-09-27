@echo off

rem cd /d %~dp1
cd

set PYTHON=".\python.exe"
echo %PYTHON% 


echo --------------------------------------------------------------
echo configure _internal env based on python-3.8.0 embed
echo --------------------------------------------------------------


if not exist python-3.8.10 (echo Error: target folder not found && PAUSE && exit /b 99)

cd python-3.8.10
cd

echo "Installing pip and so on..."
copy ..\get-pip.py
copy ..\python38._pth

rem %PYTHON% get-pip.py

echo "Installing site-packages..."
%PYTHON% -m pip install -r ..\requirements.txt

echo "type any to DONE!"
PAUSE
exit /b