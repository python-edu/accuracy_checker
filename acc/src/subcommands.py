import pandas as pd
import rasterio
import geopandas as gpd
import numpy as np
from pathlib import Path
from rasterio.features import geometry_mask

# local imports
from acc.src import cross_matrix as crm
from acc.src import binary_acc
from acc.src import functions as fn
from acc.src import clip_rasterize as clp

# --- dla podpowiadacza:
# breakpoint()
# ---


class BinaryMatrix:
    """A class for passing matrices only `binary cross matrix` - used to
    uniformly return results from `from...` functions"""
    pass


def create_binary_matrix(cross: pd.DataFrame):
    binary_obj = BinaryMatrix()

    binary = binary_acc.BinTable()
    binary_obj.binary_cross = binary(cross)

    # w układzie pionowym - na potrzeby raportu
    binary_obj.binary_cross_rep = binary(cross).T

    return binary_obj
# ---


def from_raw(args):
    """Performs calculations on data that is read from a file containing
    2 or 3 columns (raw data)."""
    kwargs = {'header': 0, 'index_col': None}
    data = pd.read_csv(args.path, sep=args.sep, **kwargs)
    raw_obj = crm.RawData(data, map_labels=args.map_labels)
    cm_obj = crm.CrossMatrix(raw_obj.map_labels,
                             raw_obj.true_values,
                             raw_obj.predicted
                             )
    valid_cm = crm.CrossMatrixValidator(cm_obj.cross_full, cm_obj.map_labels)
    # binary_cross, binary_cross_rep = create_binary_matrix(valid_cm.cross)
    binary_obj = create_binary_matrix(valid_cm.cross)
    return valid_cm, binary_obj
# ---


def from_cross_full(args):
    """Performs calculations on data of type `cross_full` - cross matrix
    with row and column descriptions and with row and column sums."""
    kwargs = {'header': 0, 'index_col': 0}
    cross_full = pd.read_csv(args.path, sep=args.sep, **kwargs)
    # cross = cross_full.iloc[:-1, :-1]
    # binary_cross, binary_cross_rep = create_binary_matrix(cross)
    
    valid_cm = crm.CrossMatrixValidator(cross_full, args.map_labels)
    binary_obj = create_binary_matrix(valid_cm.cross)
    return valid_cm, binary_obj
# binary_cross, binary_cross_rep = create_binary_matrix(valid_cm.cross)
#    return valid_cm, binary_cross, binary_cross_rep
# ---


def from_cross(args):
    """Performs calculations on `cross` data - cross matrix with row
    and column descriptions but without the sum of rows and columns."""
    kwargs = {'header': 0, 'index_col': 0}
    cross = pd.read_csv(args.path, sep=args.sep, **kwargs)
    
    valid_cm = crm.CrossMatrixValidator(cross, args.map_labels)
    binary_obj = create_binary_matrix(valid_cm.cross)
    return valid_cm, binary_obj
# ---


def from_cross_raw(args):
    """Performs calculations on `cross_raw` data - cross matrix without
    row and column descriptions, without row and column sums, i.e. a
    table of numbers only.
    It must be a square matrix, otherwise it is impossible to map classes
    (it is not known which is which)!!!
    """
    kwargs = {'header': None, 'index_col': None}
    cross_raw = pd.read_csv(args.path, sep=args.sep, **kwargs)

    valid_cm = crm.CrossMatrixValidator(cross_raw, args.map_labels)
    binary_obj = create_binary_matrix(valid_cm.cross)
    return valid_cm, binary_obj
# ---


def from_binary(args):
    """Performs calculations on data of type `binary_cross` - binary
    cross matrix containing columns (vertical layout) or rows (horizontal
    layout): TP, TN, FP, FN.
     - binary_cross_rep: in vertical format, for reporting purposes!!
    """
    kwargs = {'header': 0, 'index_col': 0}
    binary_obj = BinaryMatrix()
    valid_cm = BinaryMatrix()

    if hasattr(args, 'reversed') and args.reversed:
        binary_cross_rep = pd.read_csv(args.path, sep=args.sep, **kwargs)
        binary_cross = binary_cross_rep.T
    else:
        binary_cross = pd.read_csv(args.path, sep=args.sep, **kwargs)
        binary_cross_rep = binary_cross.T

    binary_obj.binary_cross = binary_cross
    binary_obj.binary_cross_rep = binary_cross_rep

    for attr in ['cross_raw', 'cross', 'cross_sq', 'cross_full']:
        setattr(valid_cm, attr, None)
    return valid_cm, binary_obj
# ---


def from_imgs(args):
    """Performs calculations on data of the type: reference image
    (vector) with true values and image after classification.
    Images of type `tif/geotif`, vector of type `shp/gpkg`."""

    # Loads reference data, which can be:
    #  - geopandas.GeoDataFrame, None
    #  - raster array, transform
    ref_data = clp.load_reference_data(args.path)

    # if reference is geopandas
    suffix = Path(args.path).suffix
    if suffix in ('.shp', '.gpkg'):
        clipped_image, clipped_meta = clp.clip_raster(args.path2, ref_data)
        shapes = clp.get_shapes(ref_data)
        ref_data = clp.rasterize_vector(shapes, clipped_meta)

    img = clp.load_classification_data(args.path2)

    # pd.df
    raw_data = create_raw_data(ref_data, img)

    # clean raw_data
    raw_obj = crm.RawData(raw_data, map_labels=args.map_labels)

    # create cross matrix
    cm_obj = crm.CrossMatrix(raw_obj.map_labels,
                             raw_obj.true_values,
                             raw_obj.predicted
                             )

    valid_cm = crm.CrossMatrixValidator(cm_obj.cross_full)
    binary_obj = create_binary_matrix(valid_cm.cross)
    return valid_cm, binary_obj
# ---


def create_raw_data(ref_data, class_data):
    """
    Tworzy DataFrame porównujący piksele referencyjne z klasyfikacją.
    """
    if ref_data.shape != class_data.shape:
        raise ValueError("Rozmiary danych referencyjnych i klasyfikacji nie pasują do siebie.")

    true_pixels = ref_data.flatten()
    predicted_pixels = class_data.flatten()

    df = pd.DataFrame({
        'true': true_pixels,
        'predicted': predicted_pixels
    })

    return df
