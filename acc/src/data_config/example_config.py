"""
Moduł na zawiera skrypt który:
 - uruchamiany jest przy starcie main.py lub app.py
 - kopiuje dane przykładowe na dysk
 - ustawia zmienną środowiskową do danych EXAMPLE_DATA
"""

import os
import shutil
from pathlib import Path
from importlib.resources import files
from platformdirs import user_data_dir


def get_example_data_dir(appname: str = "accuracy") -> Path:
    """Zwraca katalog na dane użytkownika zgodnie z OS, do którego zostaną
    przekopiowane dane przykładowe zawarte w pakiecie. Zwykle to:
    - linux / mac:  /home/.cache/share/accuracy/example_data/
    - windows :     %LOCALAPPDATA%\accuracy\example_data\
    """
    data_dir = Path(user_data_dir(appname)) / "example_data"
    return str(data_dir)


def copy_example_data():
    """Kopiuje pliki tylko jeśli nie istnieją.
       EXAMPLE_DATA musi być ustawione wcześniej.
    """
    dst_root = os.getenv("EXAMPLE_DATA")
    if dst_root is None:
        raise RuntimeError("EXAMPLE_DATA is not set before copy_example_data()")

    dst_pth = Path(dst_root)
    src = files("acc") / "example" / "data"

    for file in src.iterdir():
        dst = dst_pth / file.name
        if not dst.exists():
            shutil.copy(file, dst)


def config_data():
    """Konfiguruje lokalizację danych:
        - jeśli EXAMPLE_DATA istnieje - korzysta z niej
        - jeśli nie istnieje ustawia domyślną wartość.
    """
    dir_pth = os.getenv("EXAMPLE_DATA")

    # jeśli nie istnieje zmienna "EXAMPLE_DATA"
    if dir_pth is None:
        dir_pth = Path(get_example_data_dir()).resolve()
        dir_pth.mkdir(parents=True, exist_ok=True)
        os.environ["EXAMPLE_DATA"] = str(dir_pth)
    else:
        dir_pth = Path(dir_pth).resolve()

    copy_example_data()
    return
