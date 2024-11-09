import pandas as pd

from acc.src import cross_matrix
from acc.src import binary_acc
from acc.src import functions as fn

# --- dla podpowiadacza:
# breakpoint()
# ---


def create_binary_matrix(cross: pd.DataFrame):
    binary = binary_acc.BinTable()
    binary_cross = binary(cross)

    # w układzie pionowym - na potrzeby raportu
    binary_cross_rep = binary_cross.T
    return binary_cross, binary_cross_rep
# ---


def from_raw(args):
    """Performs calculations on data that is read from a file containing
    2 or 3 columns (raw data)."""
    data = pd.read_csv(args.path, sep=args.sep)
    cr = cross_matrix.ConfusionMatrix(data)
    cross = cr.cross
    cross_full = cr.cross_full

    binary_cross, binary_cross_rep = create_binary_matrix(cross)

    # either cross or binary_cros is used to calculate accuracy metrics
    # - therefore data=cross
    data = cross.copy()
    return data, cross, cross_full, binary_cross, binary_cross_rep
# ---


def from_cross_full(args):
    """Performs calculations on data of type `cross_full` - cross matrix
    with row and column descriptions and with row and column sums."""
    cross_full = pd.read_csv(args.path, sep=args.sep, index_col=0)
    cross = cross_full.iloc[:-1, :-1]
    binary_cross, binary_cross_rep = create_binary_matrix(cross)
    
    # either cross or binary_cros is used to calculate accuracy metrics
    # - therefore data=cross
    data = cross.copy()
    return data, cross, cross_full, binary_cross, binary_cross_rep
# ---


def from_cross(args):
    """Performs calculations on `cross` data - cross matrix with row
    and column descriptions but without the sum of rows and columns."""
    cross = pd.read_csv(args.path, sep=args.sep, index_col=0)
    cross_full = fn.sum_rows_cols(cross)
    binary_cross, binary_cross_rep = create_binary_matrix(cross)
    
    # either cross or binary_cros is used to calculate accuracy metrics
    # - therefore data=cross
    data = cross.copy()
    return data, cross, cross_full, binary_cross, binary_cross_rep
# ---


def from_cross_raw(args):
    """Performs calculations on `cross_raw` data - cross matrix without
    row and column descriptions, without row and column sums, i.e. a
    table of numbers only."""
    cross_raw = pd.read_csv(args.path,
                            sep=args.sep,
                            index_col=False, header=None)
    nazwy = fn.nazwij_klasy(cross_raw.shape)
    cross = cross_raw.copy()
    cross.columns = nazwy
    cross.index = nazwy
    cross_full = fn.sum_rows_cols(cross)
    binary_cross, binary_cross_rep = create_binary_matrix(cross)
    
    # either cross or binary_cros is used to calculate accuracy metrics
    # - therefore data=cross
    data = cross.copy()
    return data, cross, cross_full, binary_cross, binary_cross_rep
# ---


# def from_cross_old(args):
#     if args.data_type == "full":
#         cross_full = pd.read_csv(args.path, sep=args.sep, index_col=0)
#         cross = cross_full.iloc[:-1, :-1]
# 
#     elif args.data_type == "cross":
#         cross = pd.read_csv(args.path, sep=args.sep, index_col=0)
#         cross_full = fn.sum_rows_cols(cross)
# 
#     elif args.data_type == "raw":
#         cross_raw = pd.read_csv(args.path,
#                                 sep=args.sep,
#                                 index_col=False, header=None)
#         nazwy = fn.nazwij_klasy(cross_raw.shape)
#         cross = cross_raw.copy()
#         cross.columns = nazwy
#         cross.index = nazwy
#         cross_full = fn.sum_rows_cols(cross)
# 
#     bin = binary_acc.BinTable()
#     bin_cross = bin(cross)
# 
#     # w układzie pionowym - na potrzeby raportu
#     bin_cross_rep = bin_cross.T
#     data = None
#     return data, cross, cross_full, bin_cross, bin_cross_rep
# ---


def from_binary(args):
    """Performs calculations on data of type `binary_cross` - binary
    cross matrix containing columns (vertical layout) or rows (horizontal
    layout): TP, TN, FP, FN.
     - binary_cross_rep: in vertical format, for reporting purposes!!
    """
    
    if args.reversed:
        binary_cross_rep = pd.read_csv(args.path, sep=args.sep, index_col=0)
        binary_cross = binary_cross_rep.T
    else:
        binary_cross = pd.read_csv(args.path, sep=args.sep, index_col=0)
        binary_cross_rep = binary_cross.T

    # either cross or binary_cross is used to calculate accuracy metrics
    # - therefore data=binary_cross
    data = binary_cross.copy()
    # return data, cross, cross_full, binary_cross, binary_cross_rep
    return data, None, None, binary_cross, binary_cross_rep
# ---


def from_imgs(args):
    """Performs calculations on data of the type: reference image
    (vector) with true values and image after classification.
    Images of type `tif/geotif`, vector of type `shp/gpkg`."""
    print("\n\t", from_imgs.__name__)
# ---
