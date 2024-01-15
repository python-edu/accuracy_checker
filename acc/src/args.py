# -*- coding: utf-8-*-
"""
Plik przeznaczony dla skryptu, który pobiera z Open Hub-a informacje o
dostępnych obrazach ale nie pobiera obrazów!!!

Zawiera funkce parsującą argumenty linii poleceń.
"""

import argparse
import textwrap
from pathlib import Path

# my modules import
from simpleutils.src import argspk as apk

# local imports


# --


description = """
Skrypt sprawdza jakość klasyfikacji:
   - tworzy lub wykorzystuje istniejącą cross matrix
   - oblicza metryki dokładności."""

info = """
Metryki proste:
   - acc:  accuracy
   - ppv:  precision or positive predictive value
   - tpr:  sensitivity, recall, hit rate, true positive rate
   - tnr:  specificity, selectivity or true negative rate
   - npv:  negative predictive value
   - fnr:  miss rate or false negative rate
   - fpr:  fall-out or false positive rate
   - fdr:  false discovery rate
   - foRate: false omission rate
   - ts:  Threat score (TS) or critical success index (CSI)
   - mcc: Matthews correlation coefficient, od -1 do 1


Metryki złozone:
   - pt:  prevalence threshold
   - ba:  balanced accuracy
   - f1:  harmonic mean of precision and sensitivity
   - fm:  Fowlkes–Mallows index
   - bm:  Fowlkes–Mallows index
   - mk:  markedness or deltaP
"""


def mapa_args():
    k1 = ['input',]
    v1 = ['input']

    k2 = ['data_type', 'precision', 'revers', 'sep', 'sums']
    v2 = ['-d', '-p', '-r', '-sp', '-ss']

    k3 = ['out_dir', 'save', 'raport', 'ref', 'full_save']
    v3 = ['-o', '-s', '-rap', '-rf', '-f']

    keys = [*k1, *k2, *k3]
    vals = [*v1, *v2, *v3]
    return dict(zip(keys, vals))
# --


apk.mapa = mapa_args()

# --


def parsuj_argumenty():
    '''
    '''

    parser = apk.MyParserWithDefaults(
            formatter_class=apk.MyHelpFormatter,
            description=description,
            fromfile_prefix_chars='@',
            )

    parser.convert_arg_line_to_args = apk.convert_arg_line

    txt = '''\
    Adres pliku 'csv' z danymi. Dany mogą być:
      1. Surowe dane - przynajmniej dwie kolumny:
         -------------------------
         |    true   | predicted |
         | --------- | --------- |
         |        1  |     2     |
         |        5  |     3     |
         -------------------------

         lub

         ------------------------------------
         | etykieta |    true   | predicted |
         | -------- | --------- | --------- |
         |  trawa   |        1  |     2     |
         |   woda   |        5  |     3     |
         ------------------------------------

         gdzie:
           - 'predicted' - wynik klasyfikacji, np. 5
           - 'true'      - prawdziwa etykieta klasy np. 7.

      2. Cross matrix - gotowa tabela z opisami kolumn/wierszy i sumami.
      3. binTF - tabela TP, TN, FP, FN.'''
    parser.add_argument('input', type=str, help=textwrap.dedent(txt))

    # ---

    txt = '''\
        Mówi czym są dane wejściowe. Możliwości:
          - 'data' - 2 lub 3 kolumny w pliku csv
          - 'cros_raw' - cros matrix bez opisów wierszy i kolumn - same liczby
          - 'cros' - cros matrix z opisami wierszy i kolumn, bez sum
          - 'cros_full' - cros matrix z opisami kolumn i wierszy i z sumami
            wierszy i kolumn
          - 'bin' binTF.'''
    parser.add_argument('-d', '--data_type', type=str,
                        help=textwrap.dedent(txt), default='data')
    # -------------------------------------------------------------------------

    txt = '''Wyświetla informacje o obliczanych statystykach.'''
    parser.add_argument('-i', '--info', help=txt, action='store_true')

    # -------------------------------------------------------------------------

    txt = '''Dokładność - liczba miejsca po przecinku.'''
    parser.add_argument('-p', '--precision', type=int, help=txt, default=4)

    txt = 'Gdy cross matrix to dane wejściowe, to flaga wskazuje, że układ' \
        ' cross matrix jest odwrócony tzn. w kolumnach' \
        ' są dane referencyjne (true) a w wierszach są wyniki klasyfikacji' \
        ' obrazu (predict).'
    parser.add_argument('-r', '--revers', action='store_true', help=txt)

    txt = 'Dotyczy danych typu `crossmatrix`: zaznacz jeśli dane zawierają ' \
        ' podsumowaie wierszy i kolumn.'
    parser.add_argument('-ss', '--sums', action='store_true', help=txt)

    # -------------------------------------------------------------------------

    txt = "Określa separator kolumn pliku csv."
    parser.add_argument('-sp', '--sep', type=str, help=txt, default=';')

    txt = "Str, nazwa katalogu do zapisu danych." \
          " Katalog tworzony jest w katalogu roboczym, czyli nadrzędnym" \
          " do katalogu z danymi wejściowymi."
    parser.add_argument('-o', '--out_dir', type=str,
                        help=txt, default='results')

    txt = '''Domyślnie skrypt wyświetla wyniki na ekranie. Ta opcja powoduje,
    zapisanie wyników do osobnych plików csv:
      - cros.csv,
      - binary_cros.csv,
      - classic_acc.csv,
      - modern1.csv, modern2.csv.
               '''
    parser.add_argument('-s', '--save', help=txt, action='store_true')

    txt = '''Generuje raport w html: wszystkie tabele w jednym pliku html:
    - raport.html.
    '''
    parser.add_argument('-rap', '--raport', help=txt, action='store_true')

    # -------------------------------------------------------------------------

    txt = '''Adres pliku 'csv' z danymi referencyjnymi - 2 kolumny:
    - 'label;name'.
        '''
    parser.add_argument('-rf', '--ref', type=str, help=txt, default=None)

    txt = "Wskazuje, że wynikiem skryptyu ma być tylko raport.html. " \
        "Domyślnie oprócz raportu generuje również pliki csv."
    parser.add_argument('-f', '--full_save', help=txt, action='store_true')

    # -------------------------------------------------------------------------

    txt = "Wyświetla bardziej szczegółowe informacje."
    parser.add_argument('-v', '--verbose', help=txt, action='store_true')

    args = parser.parse_args()

    return args
# --


def validuj_args(args):
    '''Funkcja przetwarza argumenty wejściowe skryptu'''
    # nazwy katalogów i plików wyjściowych

    if not args.info:
        if args.input:
            if Path(args.input).expanduser().is_absolute():
                args.input = Path(args.input).expanduser()
            else:
                args.input = Path(args.input).resolve()

            args.work_dir = str(args.input.parent)
            args.input = str(args.input)

        if args.raport or args.save:
            args.out_dir = Path(args.work_dir) / args.out_dir

            name = Path(args.input).name.split('.')[0]
            name = f'{name}_raport.html'
            args.raport = str(args.out_dir / name)
            args.out_dir = str(args.out_dir)

        if args.ref is not None:
            args.ref = str(Path(args.ref).resolve())

    return args
# --


# --
