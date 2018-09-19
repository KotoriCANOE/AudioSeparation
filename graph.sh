postfix=7
python graph.py --postfix $postfix
python freeze_graph.py --input_binary False --input_graph model$postfix.tmp/model.graphdef --input_checkpoint model$postfix.tmp/model --output_graph model$postfix.tmp/model.pb --output_node_names Output

exit
