call E:\\DataSetAnima\\sd-scripts\venv\Scripts\activate.bat
accelerate launch --num_cpu_threads_per_process 1 E:\\DataSetAnima\\sd-scripts\\anima_train_network.py --config_file="%~dp0config.toml"
pause
