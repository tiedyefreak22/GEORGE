@echo off
cd\
cd C:\Users\khard\OneDrive\Documents\GitHub\GEORGE
call mamba activate tf2
call jupyter lab --NotebookApp.iopub_data_rate_limit=1.0e10 --NotebookApp.iopub_msg_rate_limit=1.0e7 --ServerApp.iopub_data_rate_limit=1.0e10 --ServerApp.iopub_msg_rate_limit=1.0e7
::PAUSE