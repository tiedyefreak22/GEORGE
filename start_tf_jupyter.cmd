@echo off
cd\
cd C:\Users\khard\OneDrive\Documents\GitHub\GEORGE
call mamba activate tensorflow
call jupyter lab --NotebookApp.iopub_data_rate_limit=1.0e10
::PAUSE