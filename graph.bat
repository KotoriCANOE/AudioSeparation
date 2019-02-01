cd /d "%~dp0"

FOR %%i IN (13) DO (
	python graph.py --postfix %%i
	python freeze_graph.py --input_graph model%%i.tmp\model.graphdef --input_checkpoint model%%i.tmp\model --output_graph model%%i.tmp\model.pb --output_node_names Output
)

pause

FOR %%i IN (2) DO (
	python graph.py --postfix %%i --model-file model_0210000
	python freeze_graph.py --input_graph model%%i.tmp\model.graphdef --input_checkpoint model%%i.tmp\model --output_graph model%%i.tmp\model.pb --output_node_names Output
)
