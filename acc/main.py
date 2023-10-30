# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# my modules import
from simpleutils.src import verbosepk as vb
from acc_pack.src import metryki

# local imports
from acc.src import args as parser  # parser argumentów w osobnym pliku
from acc.src import read_data  # funkcje do odczytu różnych danych
from acc.src.raport_base import SimpleRaport

# -----------------------------------------------------------------------------


class AccRaport(SimpleRaport):

    def _konvertujDane(self):
        ''' Dane w tym skrypcie to pd.DataFrame. Przeciążenie oryginalnej
        metody polega na konwersji dataFrame do html-a.
        Input:
            - self.data:    lista danych, lista [pd.DataFrame, pd.DataFrame,..]
        '''
        htmlData = [dane.to_html() for dane in self.data]
        dataToRender = dict(zip(self.opis, htmlData))
        return dataToRender
# ---


def acc_from_cros(data, args):
    """
    Oblicza wskaźniki dokładności na podstawie crossmatrix lub binary cros.
    Oblicza tradycyjne dla teledetekcji wskaźniki.
    Args:
      - data:  - cross matrix (cros), bez podsumowań wierszy i kolumn!!!
               - binary cros matrix (bin_cros)
      - args:  obiekt z atrybutami, zwykle namespase z argparse
    """
    if args.data_type in ['data', 'cros', 'cros_raw', 'cros_full']:
        acc = metryki.AccClasic(data, args.precision)

    else:
        acc = metryki.AccClasicBin(data, args.precision)

    classic_acc = acc.tabela

    return classic_acc

# ---


def acc_from_bin_cros(data, args):
    """
    Oblicza wskaźniki dokładności na podstawie binary cros.
    Oblicza wskaźniki stosowane w maszynowym uczeniu.
    Args:
      - data:  binary cros matrix (bin_cros)
      - args:  obiekt z atrybutami, zwykle namespase z argparse
    """

    acc = metryki.AccIndex(data, precision=args.precision)
    modern1 = {}
    modern2 = {}

    for k, v in vars(acc).items():
        if k in ['acc', 'ppv', 'tpr', 'tnr', 'npv', 'fnr', 'fpr', 'fdr',
                 'foRate', 'ts', 'mcc']:
            modern1[k] = v
        elif k in ['pt', 'ba', 'f1', 'fm', 'bm', 'mk']:
            modern2[k] = v

    modern1 = pd.DataFrame(modern1)
    modern2 = pd.DataFrame(modern2)

    return modern1, modern2

# ---


def main():

    # =========================================================================
    # 1. Parsowanie argumentów wejścia
    # =========================================================================
    args = parser.parsuj_argumenty()
    args = parser.validuj_args(args)
    vb.verbose(pars=True, **vars(args))

    # =========================================================================
    # 2. Odczyt danych wejściowych
    # =========================================================================
    all_data = read_data.read_data(args)
    data, cros, cros_full, binary_cros, binary_cros1 = all_data
    vb.verbose(verbose=args.verbose, data=data, cros=cros, cros_full=cros_full,
               binary_cros=binary_cros, binary_cros1=binary_cros1)

    # =========================================================================
    # 3. Tradycyjne, klasyczne wskaźniki dokładności
    # =========================================================================
    classic_acc = acc_from_cros(cros, args)
    vb.verbose(verbose=True, classic_acc=classic_acc)

    # =========================================================================
    # 4. Nowe wskaźniki dokładności
    # =========================================================================
    modern1, modern2 = acc_from_bin_cros(binary_cros, args)
    vb.verbose(verbose=True, modern1=modern1, modern2=modern2)

    # =========================================================================
    # 4.1. Liczy średnie wartości wskaźników 'modern1' i 'modern2'
    # =========================================================================

    m1 = np.round(modern1.mean(), 4)
    m2 = np.round(modern2.mean(), 4)

    modern_mean = pd.DataFrame(pd.concat([m1, m2]))
    modern_mean.columns = ['Value']

    vb.verbose(verbose=True, modern_mean=modern_mean)

    # =========================================================================
    # 5. Zapisywanie danych
    # =========================================================================
    names = ["cros", "binary_cros", "classic_acc", "modern1", "modern2"]

    if args.save:
        pths = [args.out_dir / f'{n}.csv' for n in names]
        pths = [str(p) for p in pths]
        args.out_dir.mkdir(exist_ok=True)

        vb.verbose(verbose=True, zapis="""\tZapisywanie plików `csv`:\n""")
        zapisano = []

        for i, nazwa in enumerate(names):
            if nazwa == 'cros':
                nazwa = 'cros_full'

            if nazwa == "binary_cros":
                nazwa = "binary_cros1"

            data = locals()[nazwa]
            ad = pths[i]
            data.to_csv(ad, sep=args.sep)
            zapisano.append(ad)
        vb.verbose(verbose=True, zapisano=zapisano)

    # tworzy raport html
    if args.raport:
        vb.verbose(verbose=args.verbose,
                   raport='''\n\tTworzenie i zapis raportu:\n''')

        # dane = [locals()[nazwa] for nazwa in names1]
        dane = [cros_full, binary_cros1, classic_acc, modern1, modern2]
        raport = AccRaport(data=dane, opis=names)
        raport.saveRaport(raportAdres=args.raport)

        vb.verbose(verbose=True, raport_zapisany=args.raport)
    else:
        print('''
              Nie podano opcji '-s' (save) - raport nie zostanie wykonany!
              \n''')
# #############################################################################


if __name__ == '__main__':
    main()
    wykaz = None
