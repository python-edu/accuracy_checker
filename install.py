"""Installs or removes the accuracy and accuracy_gui scripts.
1. The script called without arguments installs the scripts.
2. The script called with argument:
  - `-u` / `--uninstall` argument: it uninstalls the scripts 
  - `-p` / `--purge` argument: it uninstalls the scripts and completely removes
     the cloned repository
"""


import os
import sys
import venv
# import platform
import subprocess
import shutil
import argparse
import tempfile
import textwrap

from pathlib import Path

# --- globalne
# if sys.platform == "darwin":
#     # macOS
# elif sys.platform.startswith("linux"):
#     # Linux
# elif sys.platform == "win32":
#     # Windows
SYSTEM = sys.platform
ROOT = Path(__file__).resolve().parent
ENV_DIR = ROOT / 'env'
REQUIREMENTS = ROOT / 'requirements.txt'
# TOML = ROOT / "pyproject.toml"
ACCURACY = 'accuracy'
ACCURACYGUI = 'accuracy_gui'
PATH = os.getenv('PATH', '')

if SYSTEM == 'win32':
    import winreg as wr
    PYTHON = 'Scripts/python.exe'
    ACCURACY = f"Scripts/{ACCURACY}.exe"
    ACCURACYGUI = f"Scripts/{ACCURACYGUI}.exe"

    # zmienna środowiskowa – będzie zapisana jako REG_EXPAND_SZ
    BIN_DIR_LITERAL = r"%USERPROFILE%\bin"

    # realna ścieżka do katalogu z plikami wrap 
    BIN_DIR = Path(os.environ["USERPROFILE"]) / "bin"
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    BLOCK = ''  # tylko dla zgodności argumentów dla windows

else:
    PYTHON = 'bin/python'
    ACCURACY = f"bin/{ACCURACY}"
    ACCURACYGUI = f"bin/{ACCURACYGUI}"
    BIN_DIR = Path.home() / ".local" / "bin"
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    BLOCK = (
        "\n\n# >>> start acc/install.py PATH >>>\n"
        'export PATH="$HOME/.local/bin:$PATH"\n'
        "# <<< end acc/install.py PATH <<<\n"
    )


PYTHON = (ENV_DIR / PYTHON).resolve()

# pełna ścieżka do zainstalowanego skryptu
ACCURACY = (ENV_DIR / ACCURACY).resolve() 
ACCURACYGUI = (ENV_DIR / ACCURACYGUI).resolve() 

# ścieżki wrapperów + ich zawartość (różne dla Windows/Posix)
if SYSTEM == 'win32':
    WRAP_CLI = BIN_DIR / 'accuracy.cmd'
    WRAP_GUI = BIN_DIR / 'accuracy_gui.cmd'
    SH_CLI = f'@echo off\r\n"{ACCURACY}" %*\r\n'
    SH_GUI = f'@echo off\r\n"{ACCURACYGUI}" %*\r\n'
else:
    WRAP_CLI = BIN_DIR / 'accuracy'
    WRAP_GUI = BIN_DIR / 'accuracy_gui'
    SH_CLI = f'#!/usr/bin/env sh\nexec "{ACCURACY}" "$@"\n'
    SH_GUI = f'#!/usr/bin/env sh\nexec "{ACCURACYGUI}" "$@"\n'

# Wybór plików `rc` użytkownika
if SYSTEM == "darwin":    # macOS (zsh)
    RC_FILES = [Path.home()/".zprofile", Path.home()/".zshrc"]

elif SYSTEM == "linux":   # Linux (bash)
    RC_FILES = [Path.home()/".profile", Path.home()/".bashrc"]

else:
    RC_FILES = []  # tylko dla zgodności argumentów dla windows



# --- funkcje - parser agrumentów


def get_args():
    parser = argparse.ArgumentParser(
            prog='Install / uninstall.',
            description=textwrap.dedent(__doc__).strip(),
            formatter_class=argparse.RawDescriptionHelpFormatter
            )

    
    parser.add_argument('-u', '--uninstall', action='store_true',
                        help=("Uninstall: Removes script wrappers/bin"
                        "directories (if empty), removes the `env` environment."
                        "Leaves the repository in its cloned state."))

    parser.add_argument('-p', '--purge', action='store_true',
                        help=("Removes the script: The same as the "
                        "`--uninstall` option, plus completely removes the "
                        "repository directory from disk."))

    return parser.parse_args()

# --- funkcje instalatora


def win_add_to_path() -> bool:
    """
    dla Windows:
      - dopisuje %USERPROFILE%\\bin (BIN_DIR_LITERAL) do PATHx
        użytkownika (HKCU\\Environment).
    Zakłada istnienie:
      - BIN_DIR_LITERAL (str)
      - BIN_DIR (Path)
    Zwraca True, jeśli PATH został zmieniony, False jeśli wpis już był.
    """
    # otwórz klucz PATH użytkownika
    with wr.OpenKey(wr.HKEY_CURRENT_USER,
                    r"Environment",
                    0,
                    wr.KEY_READ | wr.KEY_SET_VALUE) as k:
        try:
            current, _typ = wr.QueryValueEx(k, "Path")
        except FileNotFoundError:
            current, _typ = "", wr.REG_EXPAND_SZ

        parts = [seg for seg in current.split(";") if seg]

        literal  = BIN_DIR_LITERAL      # np. "%USERPROFILE%\\bin"
        expanded = str(BIN_DIR)         # np. "C:\\Users\\pk\\bin"

        if literal in parts or expanded in parts:
            return False  # już jest w PATH

        new_value = (current + ";" if current else "") + literal
        wr.SetValueEx(k, "Path", 0, wr.REG_EXPAND_SZ, new_value)
        return True


def create_wrapper_files():
    # zapis wrapperów do katalogów BIN_DIR/BIN_DIR
    WRAP_CLI.write_text(SH_CLI, encoding='utf-8', newline='\n')
    WRAP_GUI.write_text(SH_GUI, encoding='utf-8', newline='\n')
    # na POSIX wrappery muszą być wykonywalne
    if SYSTEM != 'win32':
        WRAP_CLI.chmod(0o755)
        WRAP_GUI.chmod(0o755)


def add_path_to_rcfiles():
    """Dodaje wpis do plików rc (np. .bashrc). Wpis dodaje ścieżkę do katalogu
    ~/.local/bin do PATH.
    """
    txt = r".local/bin"
    if txt in PATH:
        print(f"  - {txt} already present in PATH!")
        return
    
    for rc in RC_FILES:
        rc = Path(rc)
        try:
            content = rc.read_text(encoding="utf-8")
        except FileNotFoundError:
            content = ""

        if 'export PATH="$HOME/.local/bin:$PATH"' not in content: 
            content += BLOCK
        rc.write_text(content, encoding="utf-8")
        print(f"  - {txt} was added to {rc.name}")
    return True


def install_scripts():
    print(f"Start installation:")
    print(f"  - operating system: {SYSTEM}")
    print(f"  - environment: {ENV_DIR}\n")
    # force create env: czy istnieje czy nie 
    if ENV_DIR.exists():
        shutil.rmtree(str(ENV_DIR))

    venv.EnvBuilder(system_site_packages=False,
                    with_pip=True,
                    prompt='acc',
                    upgrade_deps=True,
                    ).create(str(ENV_DIR))
    print("  - virtual env is created\n")
    if PYTHON.is_file():
        print(f"python interpreter: {PYTHON}")
    else:
        print(f"Err no python interpreter: {PYTHON}")
        sys.exit(1)

    # instalacja requirements.txt
    subprocess.run(
            [str(PYTHON), '-m', 'pip', 'install', '-r', str(REQUIREMENTS)],
            cwd=str(ROOT),
            check=True)

    print(f"  - the `requirements.txt` was installed successfully")

    # instalacja skryptów
    subprocess.run([str(PYTHON), '-m', 'pip', 'install', '.'],
                   cwd=str(ROOT),
                   check=True)

    res = subprocess.run([str(ACCURACY), '-h'], capture_output=True, text=True) 
    if res.returncode == 0:
        print(f"\n  - the {ACCURACY} was installed successfully")
    else:
        print(f"\n  - error - the {ACCURACY} was not installed")

    res = subprocess.run([str(ACCURACYGUI), '-h'],
                         capture_output=True,
                         text=True) 
    if res.returncode == 0:
        print(f"  - the {ACCURACYGUI} was installed successfully")
    else:
        print(f"  - error - the {ACCURACYGUI} was not installed")


    # dodawanie plików uruchamianych z dowolnej lokalizacji
    if SYSTEM == 'win32':
        win_add_to_path()
    else:
        BIN_DIR.mkdir(parents=True, exist_ok=True)

    print("  - ustawione katalogi i PATH")
    create_wrapper_files()
    print(f"  - utworzone wrapery:\n\t... {WRAP_CLI}\n\t... {WRAP_GUI}\n")

    # add .local/bin to PATH
    if SYSTEM != 'win32':
        add_path_to_rcfiles()


# --- funkcje deinstalatora

def uninstall_scripts():
    print("Start uninstalling scripts!")
    for file in (WRAP_CLI, WRAP_GUI):
        if file.exists():
            file.unlink()
            print(f"  - {file.name} was removed") 

    # usuwanie katalogu bin po wraperach jeśli jest pusty
    try:
        BIN_DIR.rmdir()
        print(f"  - '{BIN_DIR}' was empty and was removed!")
    except Exception:
        pass

    folders = ('env', 'built', 'dist', '__pycache__')

    for folder in folders:
        try:
            shutil.rmtree(str(ROOT / folder))
            print(f"  - removed: {ROOT / folder}")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"  - warn: cannot remove {ROOT / folder}: {e}")


# --- funkcje purge
def run_purge() -> None:
    """Copy acc/acc/src/purge.py to temp and run it detached:
        <root> <ppid> <bin_dir>."""
    src = ROOT / "acc" / "src" / "purge.py"     # ścieżka w repo
    dst = Path(tempfile.gettempdir()) / "acc_purge.py"
    shutil.copy2(src, dst)

    ppid = os.getppid()
    cmd = [sys.executable, str(dst),
           "--root", str(ROOT),
           "--ppid", str(ppid),
           "--bin_dir", str(BIN_DIR),
           "--block", BLOCK,
           "--rc_files", ";".join(map(str, RC_FILES)),
           ]

    if SYSTEM == "win32":
        DETACHED = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        subprocess.Popen(cmd, creationflags=DETACHED, close_fds=True, cwd=tempfile.gettempdir())
    else:
        subprocess.Popen(
            cmd,
            start_new_session=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            close_fds=True,
            cwd=tempfile.gettempdir(),
        )



# --- main()


def main(args):
    if args.purge:
        # uruchom purge.py w tmp dir
        run_purge()
    elif args.uninstall:
        uninstall_scripts()
    else:
        install_scripts()


if __name__ == "__main__":
    args = get_args()
    main(args)

