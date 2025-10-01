"""
Wrapper app.py:
    - dodaje funkcje główną, którą można wywołać w `pyproject.toml`
      w entry point czyli umożliwia zbudowanie nazwy, która uruchamia app.py
"""

# acc/app_entry.py
import sys
import subprocess
from pathlib import Path

def main():
    # app_entry.py i app.py są w tym samym pakiecie: acc/
    app_path = Path(__file__).with_name("app.py")

    # sys.executable: Python, który odpalono entry point (venv!)
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path),
           *sys.argv[1:]]

    # uruchom Streamlit i przekaż kod wyjścia
    try:
        res = subprocess.run(cmd)
        raise SystemExit(res.returncode)
    except KeyboardInterrupt:
        raise SystemExit(130)  # konwencjonalny kod przerwania
