"""
Single source of truth for CRS handling. Every other module should call into
this rather than reprojecting ad hoc inline -- that habit is the #1 silent-
bug source in a pipeline mixing SAR rasters (UTM), AIS points (WGS84), and
wind/current grids (geographic, 0-360 or -180/180 longitude depending on
source). Functions here fail loudly on missing/mismatched CRS rather than
silently assuming WGS84.
"""
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
import numpy as np


def get_scene_utm_crs(sar_path):
    """Read the SAR scene's native CRS (projected UTM after SNAP terrain
    correction) -- this becomes the standard CRS for the whole pipeline run.
    Raises if the input isn't already projected, which usually means
    terrain correction was skipped."""
    with rasterio.open(sar_path) as src:
        if not src.crs.is_projected:
            raise ValueError(
                f"{sar_path} is not in a projected CRS ({src.crs}). "
                "Run terrain correction (SNAP/pyroSAR) before this step -- "
                "raw SAR products are not georeferenced to a projected CRS."
            )
        return src.crs


def reproject_raster_to(src_path, dst_crs, dst_path, resampling=Resampling.bilinear):
    """Reproject a wind/current raster (or any raster) into the target CRS.
    Fails loudly if the source has no CRS at all -- common with raw NetCDF
    exports from ERA5/CMEMS, which can load into rasterio without one."""
    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError(
                f"{src_path} has no CRS attached. ERA5/CMEMS NetCDF often "
                "needs an explicit CRS assigned (EPSG:4326) before rasterio "
                "will reproject it -- check with `gdalinfo {src_path}` first, "
                "and verify longitude convention (0-360 vs -180/180) while "
                "you're there; that mismatch reprojects 'successfully' but "
                "shifts the whole field silently."
            )
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(crs=dst_crs, transform=transform, width=width, height=height)
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i), destination=rasterio.band(dst, i),
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=transform, dst_crs=dst_crs, resampling=resampling,
                )
    return dst_path


def normalize_longitude_convention(lon_array, target="-180_180"):
    """ERA5/CMEMS sometimes ship 0-360 longitude, sometimes -180/180,
    depending on product and source. Normalize explicitly before any
    spatial join -- don't assume."""
    lon_array = np.asarray(lon_array)
    if target == "-180_180":
        return np.where(lon_array > 180, lon_array - 360, lon_array)
    return np.where(lon_array < 0, lon_array + 360, lon_array)


def reproject_points_to(gdf: gpd.GeoDataFrame, dst_crs) -> gpd.GeoDataFrame:
    """AIS points (typically EPSG:4326) -> SAR scene's UTM CRS. Raises if
    the GeoDataFrame has no CRS set -- silently assuming EPSG:4326 here is
    exactly the kind of bug this module exists to prevent."""
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS set -- set it explicitly, don't assume EPSG:4326.")
    return gdf.to_crs(dst_crs)


def assert_same_crs(*objects):
    """Call this immediately before any pixel/vector overlay operation.
    Cheap insurance against the single most common bug in this pipeline.
    Accepts rasterio datasets, GeoDataFrames, or raw CRS objects."""
    crss = [getattr(obj, "crs", obj) for obj in objects]
    if len({str(c) for c in crss}) > 1:
        raise ValueError(f"CRS mismatch across inputs: {crss}")
