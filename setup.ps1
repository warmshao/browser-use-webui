deactivate

Remove-Item -Recurse -Force .venv

# Step 2: Set Up Python Environment
uv venv --python 3.11

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Step 3: Install Dependencies
uv pip install -r requirements.txt
playwright install


# Step 4: Run web ui in local
python webui.py --ip 127.0.0.1 --port 7788
Write-Output "Setup complete. Virtual environment activated."
