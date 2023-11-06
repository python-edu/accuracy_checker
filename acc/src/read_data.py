# -*- coding: utf-8 -*-

# import numpy as np
import pandas as pd
from acc_pack.src import cross_matrix
from acc_pack.src import binary_acc

# ---


def sum_rows_cols(cros: pd.DataFrame):
    """Oblicza sumy w wierszach i kolumnach dla crossmatrix oraz dodaje
    te sumy do crossmatrix jako wiersze i kolumny podsumowujące."""
    cros = cros.copy()
    sum_row = cros.sum(axis=1).to_numpy()
    cros.loc[:, 'sum_row'] = sum_row

    sum_kol = cros.sum(axis=0).to_numpy()
    cros.loc['sum_kol', :] = sum_kol

    cros = cros.astype('int')
    return cros

# --


def nazwij_klasy(shape):
    """
    Jeśli cros nie ma nazw wierszy i kolumn (cros_raw) to tworzy nazwy klas:
    - kl_01, kl_02,...
    """
    n = max(shape)
    names = [f'kl_{i:0>2}' for i in range(1, n+1)]
    return names

# ---


def read_data(args):
    data, cros, cros_full, bin_cros, bin_cros1 = [None for _ in range(5)]

    if args.data_type in ['data', 'cros_raw', 'cros', 'cros_full']:
        if args.data_type == 'data':
            data = pd.read_csv(args.input, sep=args.sep)
            cr = cross_matrix.ConfusionMatrix(data)
            cros = cr.cros
            cros_full = cr.cros_full

        elif args.data_type == 'cros_full':
            cros_full = pd.read_csv(args.input, sep=args.sep, index_col=0)

        elif args.data_type == 'cros':
            cros = pd.read_csv(args.input, sep=args.sep, index_col=0)
            if args.sums:
                cros_full = sum_rows_cols(cros)
            else:
                cros_full = cros.copy()

        elif args.data_type == 'cros_raw':
            cros_raw = pd.read_csv(args.input, sep=args.sep,
                                   index_col=False, header=None)
            nazwy = nazwij_klasy(cros_raw.shape)
            cros = cros_raw.copy()
            cros.columns = nazwy
            cros.index = nazwy

        bin = binary_acc.BinTable()
        bin_cros = bin(cros)

    else:
        bin_cros = pd.read_csv(args.input, sep=args.sep, index_col=0)
        # sprawdź w jakim układzie jest data DataFrame
        # print(f'\ntuu:\n{binTF}\n\n')
        kols = set(bin_cros.columns.to_list())
        spr = set(['TP', 'TN', 'FP', 'FN'])
        if spr.issubset(kols):
            bin_cros = bin_cros.T

    # w układzie pionowym - na potrzeby raportu
    bin_cros1 = bin_cros.T
    # toSave = ['binTFv']

    return data, cros, cros_full, bin_cros, bin_cros1
# --
