"""
Moduł na zawiera skrypt który:
 - uruchamiany jest przy starcie main.py lub app.py
 - kopiuje dane przykładowe na dysk
 - ustawia zmienną środowiskową do danych EXAMPLE_DATA
 - EXAMPLE_DATA: str - wskazuje (path) na katalog na dysku z danymi
   przykładowymi
"""

import os
import shutil
from pathlib import Path
from importlib.resources import files
from platformdirs import user_data_dir


def get_example_data_dir(appname: str = "accuracy") -> Path:
    """Zwraca katalog na dane użytkownika zgodnie z OS (EXAMPLE_DATA), do
    którego zostaną przekopiowane dane przykładowe zawarte w pakiecie. Zwykle
    to:
    - linux / mac:  /home/.cache/share/accuracy/example_data/
    - windows :     %LOCALAPPDATA%\accuracy\example_data\
    """
    data_dir = Path(user_data_dir(appname)) / "example_data"
    return str(data_dir)


def copy_example_data():
    """Kopiuje pliki tylko jeśli nie istnieją.
       EXAMPLE_DATA musi być ustawione wcześniej.
    """
    data_dir = os.getenv("EXAMPLE_DATA")
    if data_dir is None:
        raise RuntimeError("EXAMPLE_DATA is not set before copy_example_data()")

    data_dir = Path(data_dir)
    src = files("acc") / "example" / "data"

    for file in src.iterdir():
        dst = data_dir / file.name
        if not dst.exists():
            shutil.copy(file, dst)


def config_data():
    """Konfiguruje lokalizację danych:
        - jeśli EXAMPLE_DATA istnieje - korzysta z niej
        - jeśli nie istnieje ustawia domyślną wartość.
    """
    data_dir = os.getenv("EXAMPLE_DATA")

    # jeśli nie istnieje zmienna "EXAMPLE_DATA"
    if data_dir is None:
        data_dir = Path(get_example_data_dir()).resolve()
        os.environ["EXAMPLE_DATA"] = str(data_dir)
    
    data_dir = Path(data_dir).resolve()

    # if exist -> make clean
    if data_dir.exists():
        shutil.rmtree(data_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    copy_example_data()
    return
