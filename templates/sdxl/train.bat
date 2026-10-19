call E:\\StableDiffusion\\sd-scripts\venv\Scripts\activate.bat
accelerate launch --num_cpu_threads_per_process 1 E:\\StableDiffusion\\sd-scripts\\sdxl_train_network.py --config_file="E:\\DataSet\\rulutieh\\config.toml"
pause
