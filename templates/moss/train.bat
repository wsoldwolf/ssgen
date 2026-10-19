call "E:\DataSetMoss\sd-scripts\venv\Scripts\activate.bat"
accelerate launch --num_cpu_threads_per_process 1 "E:\DataSetMoss\sd-scripts\moss_image_train_network.py" --config_file="E:\DataSetMoss\arcticwolf\config.toml"
pause
