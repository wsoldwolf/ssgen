robocopy "E:\DataSetMoss\arcticwolf\LoRA\arcticwolf" "C:\Software\ComfyUI\models\loras\moss\arcticwolf" "*.safetensors" /R:60 /W:5
if errorlevel 8 exit /b %ERRORLEVEL%
exit /b 0
