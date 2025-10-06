"""
Wrapper app.py:
    - dodaje funkcje główną, którą można wywołać w `pyproject.toml`
      w entry point czyli umożliwia zbudowanie nazwy, która uruchamia app.py
"""

# acc/app_entry.py
import os
import sys
import subprocess
from pathlib import Path

LOG_DIR = os.getenv('ACCURACY_LOG_DIR', None)

def launch_streamlit(app_path, args):
    # katalog na pliki logów - istnieje w repo zawsze
    log_dir = LOG_DIR or Path(app_path).parent.parent / 'logs'
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), *args]

    try:
        has_tty = bool(sys.stdout.isatty())
    except Exception:
        has_tty = False

    kwargs = {"close_fds": True}

    if has_tty:
        # Logi w konsoli (Linux/macOS/Windows w terminalu z python.exe)
        try:
            res = subprocess.run(cmd)
            return res.returncode
        except KeyboardInterrupt:
            return 130

    # Brak TTY, odpal w tle, logi do pliku
    #  - Windows gui-scripts/pythonw.exe, launchery itp.
    log_path = Path(log_dir) / "accuracy_gui.log"
    # log_path.parent.mkdir(parents=True, exist_ok=True)

    if os.name == "nt":
        # Odłącz proces i nie pokazuj okna (gdyby jednak był python.exe)
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        CREATE_NO_WINDOW = 0x08000000
        kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True

    with open(str(log_path), "a", encoding="utf-8") as log:
        return subprocess.Popen(cmd, stdout=log, stderr=log, **kwargs)

def main():
    app_path = Path(__file__).with_name("app.py")
    launch_streamlit(app_path, sys.argv[1:])

