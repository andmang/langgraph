# Check if Python 3.12 is installed
python3.12 --version

# Create venv with Python 3.12
python3.12 -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows

# Install dependencies (after creating requirements.txt below)
pip install -r requirements.txt

python -m pip install -U ipykernel



```
# Uninstall first to be safe
pip uninstall -y pygraphviz

# Reinstall pointing explicitly to Homebrew's graphviz
export CFLAGS="-I$(brew --prefix graphviz)/include"
export LDFLAGS="-L$(brew --prefix graphviz)/lib"
pip install --no-cache-dir pygraphviz
```