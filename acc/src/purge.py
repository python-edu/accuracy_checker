"""
Script intended for internal use by the uninstaller (install.py).
Its task is to clean the system of files, directories, and configuration entries
created by install.py.

The script will be run in a separate process and from a temporary directory so
that it can delete itself.
"""

import os
import sys
import argparse
import signal
import shutil
import textwrap
import re
from pathlib import Path


# --- globalne ---
SYSTEM = sys.platform


def parse_args():
    p = argparse.ArgumentParser(
            description=textwrap.dedent(__doc__).strip(),
            formatter_class=argparse.RawDescriptionHelpFormatter
            )
    p.add_argument("--root",
                   required=True,
                   type=str,
                   help="Path to the repository directory to remove.")

    p.add_argument("--ppid",
                   type=int,
                   required=True,
                   help="PID of the parent process to terminate.")

    p.add_argument("--bin_dir",
                   type=str,
                   required=True,
                   help="Path to folder containing wrap files.")

    p.add_argument("--block",
                   type=str,
                   required=True,
                   help="Entry (text) added by the installer to the 'rc' files."
                   )

    p.add_argument("--rc_files",
                   type=str,
                   required=True,
                   help=("List of strings eg.: file;file - list of paths to "
                         "'rc' files.")
                   )
    return p.parse_args()


def process_args(args):
    args.bin_dir = Path(args.bin_dir)
    args.rc_files = [s for s in
                     (f.strip() for f in args.rc_files.split(';')) if s
                     ]
    if SYSTEM == 'win32':
        args.wrap_cli = args.bin_dir / 'accuracy.cmd'
        args.wrap_gui = args.bin_dir / 'accuracy_gui.cmd'
    else:
        args.wrap_cli = args.bin_dir / 'accuracy'
        args.wrap_gui = args.bin_dir / 'accuracy_gui'

    return args


def kill_parent(pid: int) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"[ok] sent SIGTERM to {pid}")
        return True
    except ProcessLookupError:
        print(f"[skip] process {pid} does not exist")
        return False
    except PermissionError:
        print(f"[warn] no permission to signal {pid}")
        return False
    except Exception as e:
        print(f"[warn] SIGTERM {pid} failed: {e}")
        return False


def rmtree_repo(path: Path) -> bool:
    """path: to repository directory"""
    try:
        shutil.rmtree(path)
        print(f"[ok] removed {path}")
        return True
    except FileNotFoundError:
        print(f"[skip] not found: {path}")
        return False
    except Exception as e:
        print(f"[warn] rmtree({path}) failed: {e}")
        return False


def rm_wrappers(args):
    """Usuwa wrappers:
      - pliki uruchamiające skrypty
      - katalog `bin/` o ile jest pusty.
    """
    for file in (args.wrap_cli, args.wrap_gui):
        if file.exists():
            file.unlink()
            print(f"  - {file.name} was removed")

    try:
        args.bin_dir.rmdir()
        print(f"  - '{args.bin_dir}' was empty and was removed!")
    except Exception:
        print(f"  - '{args.bin_dir}' is not empty, so it was not removed!")


def rm_from_rcfiles(args):
    """Usuwa wpisy, które dodał do plików rc: nie dotyczy Windows!!!"""
    # jeśli to windows i został wykasowany katalog bin
    if SYSTEM != 'win32' and not Path(args.bin_dir).exists():
        for file in args.rc_files:
            p = Path(file)
            try:
                txt = p.read_text(encoding="utf-8")
            except FileNotFoundError:
                print(f"[skip] rc file not found: {p}")
                continue
            except Exception as e:
                print(f"[warn] cannot read rc file {p}: {e}")
                continue

            changed = False
            if args.block in txt:
                txt = txt.replace(args.block, '')
                changed = True
            else:
                # awaryjnie: dopasowanie regex na escapowanym bloku
                if args.block and re.search(re.escape(args.block), txt):
                    txt = re.sub(re.escape(args.block), '', txt)
                    changed = True
            if changed:
                try:
                    p.write_text(txt, encoding="utf-8")
                    print(f"  - {p.name} - removed PATH entry for '.local/bin/'")
                except Exception as e:
                    print(f"[warn] cannot write rc file {p}: {e}")

def main():
    args = parse_args()
    args = process_args(args)

    # zabij / zamknij bieżący terminal
    kill_parent(args.ppid)

    # usuń repozytorium
    rmtree_repo(args.root)
    
    # usuń pliki wrapperów
    rm_wrappers(args)

    # usuń wpisy z rc_files - tylko linux
    rm_from_rcfiles(args)

    # usuń samego siebie (plik w /tmp)
    try:
        Path(__file__).unlink()
    except Exception:
        pass

if __name__ == "__main__":
    main()

