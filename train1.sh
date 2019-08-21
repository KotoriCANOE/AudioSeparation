python train.py "/data/Songs/NSH" --processes 6 --max-steps 127000 --random-seed 0 --device /gpu:0 --postfix 20

exit

python train.py "/data/Songs/NSH_npz" --packed --processes 2 --max-steps 127000 --random-seed 0 --device /gpu:0 --postfix 20
