source venv/Scripts/Activate

echo "Virtual environment activated."

# Ensure pip is installed (in case it's missing)
python -m pip install --upgrade pip

echo "Pip updated and installed."

# Install required packages from requirements.txt
pip install -r requirements.txt

# Inform the user
echo "All requirements installed."