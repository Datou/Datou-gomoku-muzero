echo "Start VENV"

call .venv\Scripts\activate.bat


tensorboard --logdir E:\Gomoku_EfficientZero\outputs\logs --host=127.0.0.1 --port=6007
pause