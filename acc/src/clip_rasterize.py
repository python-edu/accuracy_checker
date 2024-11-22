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
    Wczytuje dane referencyjne jako numpy array.
    Obsługuje formaty: TIFF, Shapefile (SHP), Geopackage (GPKG).
    """
    suffix = Path(ref_path).suffix
    if suffix in ('.tif', '.tiff', '.TIF', '.TIFF'):
        with rasterio.open(ref_path) as src:
            ref_data = np.astype(src.read(1), np.uint8)
            # ref_transform = src.transform
        ## return ref_data, ref_transform
        return ref_data
    elif ref_path.endswith('.shp') or ref_path.endswith('.gpkg'):
        gdf = gpd.read_file(ref_path)
        return gdf


def load_classification_data(class_path):
    """
    Wczytuje dane po klasyfikacji z pliku TIFF.
    """
    with rasterio.open(class_path) as src:
        class_data = src.read(1)
    return np.astype(class_data, np.uint8)


def clip_raster(raster_path: str, gdf: gpd.GeoDataFrame):
    with rasterio.open(raster_path) as src:
        # Przycięcie rastra do obszaru wektora
        shapes = gdf.geometry.values  # Geometrie wektora
        clipped_image, clipped_transform = mask(src, shapes, crop=True)
        clipped_meta = src.meta.copy()
        clipped_meta.update({
            "driver": "GTiff",
            "height": clipped_image.shape[1],
            "width": clipped_image.shape[2],
            "transform": clipped_transform,
            "dtype": "uint8"
        })

    return clipped_image, clipped_meta
    

def get_shapes(gdf):
    class_column = next((col for col in gdf.columns if gdf[col].dtype \
            in ('int', 'int16', 'int32', 'int64'))
                        )
    shapes = tuple(zip(gdf['geometry'], gdf.loc[:, class_column]))
    return shapes


def rasterize_vector(shapes, meta):
    rasterized = rasterize(
        shapes,
        out_shape=(meta['height'], meta['width']),
        transform=meta['transform'],
        fill=0,  # Wartość domyślna dla pikseli poza kształtami
        dtype=meta['dtype']
    )
    return rasterized
