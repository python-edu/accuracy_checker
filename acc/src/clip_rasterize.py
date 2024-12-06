"""
Contains functions for:
- loading raster and vector data
- cropping an image to a vector
- rasterizing a vector
"""

import rasterio
from rasterio.mask import mask
import geopandas as gpd
from rasterio.features import rasterize
import numpy as np
from pathlib import Path



def load_reference_data(ref_path):
    """
    Loads reference data as a numpy array from supported formats
    (TIFF, Shapefile, Geopackage).
    
    Args:
        ref_path (str): Path to the reference data file.
    
    Returns:
        numpy.ndarray or geopandas.GeoDataFrame: Reference data as a numpy
                                                 array for raster formats 
                                                 (TIFF) or a GeoDataFrame
                                                 for vector formats (Shapefile,
                                                 Geopackage).
    """
    suffix = Path(ref_path).suffix.lower()
    if suffix in (".tif", ".tiff"):
        with rasterio.open(ref_path) as src:
            ref_data = src.read(1).astype(np.uint8)
        return ref_data
    elif suffix in (".shp", ".gpkg"):
        return gpd.read_file(ref_path)


def load_classification_data(class_path):
    """
    Loads classification data from a TIFF file as a numpy array.
    
    Args:
        class_path (str): Path to the classification data file.
    
    Returns:
        numpy.ndarray: Classification data as a numpy array with dtype uint8.
    """
    with rasterio.open(class_path) as src:
        class_data = src.read(1).astype(np.uint8)
    return class_data


def clip_raster(raster_path: str, gdf: gpd.GeoDataFrame):
    with rasterio.open(raster_path) as src:
        # Przycięcie rastra do obszaru wektora
        shapes = gdf.geometry.values  # Geometrie wektora
        clipped_image, clipped_transform = mask(src, shapes, crop=True)
        clipped_meta = src.meta.copy()
        clipped_meta.update(
            {
                "driver": "GTiff",
                "height": clipped_image.shape[1],
                "width": clipped_image.shape[2],
                "transform": clipped_transform,
                "dtype": "uint8",
            }
        )

    return clipped_image, clipped_meta


def get_shapes(gdf):
    class_column = next(
        (
            col
            for col in gdf.columns
            if gdf[col].dtype in ("int", "int16", "int32", "int64")
        )
    )
    shapes = tuple(zip(gdf["geometry"], gdf.loc[:, class_column]))
    return shapes


def rasterize_vector(shapes, meta):
    rasterized = rasterize(
        shapes,
        out_shape=(meta["height"], meta["width"]),
        transform=meta["transform"],
        fill=0,  # Wartość domyślna dla pikseli poza kształtami
        dtype=meta["dtype"],
    )
    return rasterized
