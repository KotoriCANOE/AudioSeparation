chcp 65001
cd /d "%~dp0"

python AudioSeparation.py model7.tmp/model.pb -i "D:\CloudMusic\泠鸢yousa" -o "result"

pause
