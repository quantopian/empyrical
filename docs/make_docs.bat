@ECHO OFF
REM Makes the Sphinx documentation files

FOR %%a IN (%~dp0\.) do set SOURCE=%%~dpa
set OLD_PYTHONPATH=%PYTHONPATH%
set PYTHONPATH=%PYTHONPATH%;%SOURCE%

sphinx-apidoc -f -o . ../empyrical
./make.bat html
set PYTHONPATH=%OLD_PYTHONPATH%
