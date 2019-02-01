chcp 65001
cd /d "%~dp0"

python train.py "F:\Datasets\Songs\NSH" --processes 4 --max-steps 255000 --random-seed 0 --device /gpu:0 --batch-size 1 --postfix 15

pause

python train.py "F:\Datasets\Songs\NSH_npz" --packed --processes 2 --max-steps 127000 --random-seed 0 --device /gpu:0 --batch-size 2 --postfix 4
