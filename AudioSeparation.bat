chcp 65001
cd /d "%~dp0"

SET "DIR=F:\Datasets\Songs\Test"
SET POSTFIX=20
python AudioSeparation.py model%POSTFIX%.tmp/model.pb -i "%DIR%" -o "%DIR%\..\TestResult\%POSTFIX%"

pause
