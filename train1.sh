postfix=21
python train.py "/data/Songs/NSH_npz" --packed --processes 2 --max-steps 127000 --random-seed 0 --postfix $postfix
python graph.py --postfix $postfix
python freeze_graph.py --input_binary False --input_graph model$postfix.tmp/model.graphdef --input_checkpoint model$postfix.tmp/model --output_graph model$postfix.tmp/model.pb --output_node_names Output

exit

python train.py "/data/Songs/NSH" --processes 6 --max-steps 127000 --random-seed 0 --postfix $postfix
