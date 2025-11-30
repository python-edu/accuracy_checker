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
    python_exe = Path(sys.executable)

    # windows: jeśli gui-script wymusza pythonw.exe, podmień
    if python_exe.name.lower() == "pythonw.exe":
        python_exe = python_exe.with_name("python.exe")

    cmd = [
        str(python_exe),
        "-m", "streamlit",
        "run", str(app_path),
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
        *args
    ]

    # jeśli jesteśmy w konsoli → normalny tryb
    if sys.stdout.isatty():
        try:
            return subprocess.run(cmd).returncode
        except KeyboardInterrupt:
            return 130

    # jeśli nie ma TTY → uruchom w tle (kliknięcie)
    log_path = Path(LOG_DIR or Path(app_path).parent.parent / 'logs') / "accuracy_gui.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as log:
        if os.name == "nt":
            DETACHED = 0x00000008 | 0x00000200
            return subprocess.Popen(cmd, stdout=log, stderr=log,
                                    creationflags=DETACHED, close_fds=True)
        else:
            return subprocess.Popen(cmd, stdout=log, stderr=log,
                                    start_new_session=True, close_fds=True)


def main():
    app_path = Path(__file__).with_name("app.py")
    launch_streamlit(app_path, sys.argv[1:])

