echo "Start VENV"

call .venv\Scripts\activate.bat

python -m unittest tests/test_mcts_logic.py
pause