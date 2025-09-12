@echo off
chcp 65001
cd /d %~dp0
cls

echo - - - - - -
echo 필수 라이브러리를 설치합니다...
echo - - - - - -
timeout 1 > nul
pip install -r requirements.txt
timeout 1 > nul

echo - - - - - -
echo 메인 프로그램을 실행합니다...
echo - - - - - -
timeout 1 > nul
python main.py
timeout 1 > nul

pause