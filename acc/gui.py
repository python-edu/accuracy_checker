# import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tkinter import ttk

from acc.main import main   # to jest Twój main(), nic nie zmieniam
from acc.src.args_data import args_func as afn
from acc.src.args_data.args import parsuj_argumenty


def setup_input_file(key: int, paths: list):
    file_no = {1: {'type_name': "CSV / Raster",
                   'filters': ("*.csv", "*.tif", "*.tiff", "*.TIF", "*.TIFF")
                   },
              2: {'type_name': "Raster / vector",
                   'filters': ("*.tif", "*.tiff", "*.TIF", "*.TIFF",
                              "*.shp", "*.gpkg")
                   },
              3: {'type_name': "JSON",
                   'filters': ("*.json")
                   }
               }

    type_name = file_no[key]['type_name']
    filters = file_no[key]['filters']

    file = filedialog.askopenfilename(
            title="File",
            filetypes=[(type_name, filters)])
    
    # zmienna globalna paths w `run()`
    if file:
        paths.append(file)


def run():
    # --- KROK 1: otwórz okno TK do wyboru pliku ---
    root = Tk()
    # root = tk.Tk()
    # root.withdraw()  # ukryj główne okno

    # zmienna globalna run()
    paths = []

    b1 = ttk.Button(root, text='File',
                   command=lambda: setup_input_file(1, paths)).pack()
    b2 = ttk.Button(root, text='File',
                   command=lambda: setup_input_file(2, paths)).pack()
    b3 = ttk.Button(root, text='File',
                   command=lambda: setup_input_file(3, paths)).pack()
    b4 = ttk.Button(root, text='Run', command=root.destroy).pack()
    
    root.mainloop()
    # --- brak wyboru pliku → zakończ program ---
    if not paths:
        print("Nie wybrano pliku.")
        return

    # --- KROK 2: przygotuj args ---
    parser = parsuj_argumenty()
    args = parser.parse_args([*paths, '-v'])

    # args = SimpleNamespace(path=[filepath], verbose=True)
    args = afn.args_validation(args)

    # breakpoint()

    # --- KROK 3: uruchom Twoje main(args) ---
    # print(f"\n[accuracy_gui] Wywołuję main() z args.path = {filepath}\n")
    main(args)
