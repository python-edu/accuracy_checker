import pandas as pd
import rasterio
import geopandas as gpd
import numpy as np

from rasterio.features import geometry_mask

# local imports
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
    ref_data, ref_transform = load_reference_data(args.path)

    if isinstance(ref_data, gpd.GeoDataFrame):
        # rasterize vector data
        with rasterio.open(args.path2) as src:
            ref_data = rasterize_vector_reference(ref_data,
                                                  src.shape,
                                                  src.transform)

    # load classified image
    class_data = load_classification_data(args.path2)

    # create raw data - 2 columns: true and predicted
    raw_data = create_raw_data(ref_data, class_data)

    cr = cross_matrix.ConfusionMatrix(raw_data)
    cross = cr.cross
    cross_full = cr.cross_full

    binary_cross, binary_cross_rep = create_binary_matrix(cross)

    # either cross or binary_cros is used to calculate accuracy metrics
    # - therefore data=cross
    data = cross.copy()
    return data, cross, cross_full, binary_cross, binary_cross_rep
# ---


def load_reference_data(ref_path):
    """
    Wczytuje dane referencyjne jako numpy array.
    Obsługuje formaty: TIFF, Shapefile (SHP), Geopackage (GPKG).
    """
    if ref_path.endswith('.tif'):
        with rasterio.open(ref_path) as src:
            ref_data = np.astype(src.read(1), np.uint8)
            ref_transform = src.transform
        return ref_data, ref_transform
    elif ref_path.endswith('.shp') or ref_path.endswith('.gpkg'):
        gdf = gpd.read_file(ref_path)
        return gdf, None
# ---


def rasterize_vector_reference(vector_data, shape, transform):
    """
    Rasteryzuje dane wektorowe referencyjne, zwracając maskę numpy array.
    """
    mask = geometry_mask(
        [geom for geom in vector_data.geometry],
        out_shape=shape,
        transform=transform,
        invert=True
    )
    ref_data = np.where(mask, 1, 0)  # Przypisz etykiety klas dla każdej geometrii w razie potrzeby
    ref_data = np.astype(ref_data, np.uint8)
    return ref_data
# ---


def load_classification_data(class_path):
    """
    Wczytuje dane po klasyfikacji z pliku TIFF.
    """
    with rasterio.open(class_path) as src:
        class_data = src.read(1)
    return np.astype(class_data, np.uint8)
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
