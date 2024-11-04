import pandas as pd

from acc.src import cross_matrix
from acc.src import binary_acc
from acc.src import functions as fn

# --- dla podpowiadacza:
# breakpoint()
# ---


def from_raw(args):
    data = pd.read_csv(args.path, sep=args.sep)
    cr = cross_matrix.ConfusionMatrix(data)
    cross = cr.cross
    cross_full = cr.cross_full

    bin = binary_acc.BinTable()
    bin_cross = bin(cross)

    # w układzie pionowym - na potrzeby raportu
    bin_cross_rep = bin_cross.T
    # toSave = ['binTFv']

    return data, cross, cross_full, bin_cross, bin_cross_rep
# ---


def from_cross(args):
    if args.data_type == "full":
        cross_full = pd.read_csv(args.path, sep=args.sep, index_col=0)
        cross = cross_full.iloc[:-1, :-1]

    elif args.data_type == "cross":
        cross = pd.read_csv(args.path, sep=args.sep, index_col=0)
        cross_full = fn.sum_rows_cols(cross)

    elif args.data_type == "raw":
        cross_raw = pd.read_csv(args.path,
                                sep=args.sep,
                                index_col=False, header=None)
        nazwy = fn.nazwij_klasy(cross_raw.shape)
        cross = cross_raw.copy()
        cross.columns = nazwy
        cross.index = nazwy
        cross_full = fn.sum_rows_cols(cross)

    bin = binary_acc.BinTable()
    bin_cross = bin(cross)

    # w układzie pionowym - na potrzeby raportu
    bin_cross_rep = bin_cross.T
    data = None
    return data, cross, cross_full, bin_cross, bin_cross_rep
# ---


def from_bin(args):
    print("\n\t", from_bin.__name__)
# ---


def from_imgs(args):
    print("\n\t", from_imgs.__name__)
# ---
