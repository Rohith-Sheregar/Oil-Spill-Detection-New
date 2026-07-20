"""
NOAA Marine Cadastre AIS data: free, bulk, no login.
https://hub.marinecadastre.gov/pages/vesseltraffic -> download by year/UTM zone
(the Gulf of Mexico AOI typically spans NOAA zones 15-16 depending on exact
longitude -- check the zone map before downloading).

Files arrive as monthly CSVs, one row per AIS position report. They are
LARGE -- a single Gulf-zone month can be several GB and tens of millions of
rows. Never `pd.read_csv` the whole file blindly; filter by bbox/time while
reading in chunks, as below.

Columns (current NOAA schema): MMSI, BaseDateTime, LAT, LON, SOG, COG,
Heading, VesselName, IMO, CallSign, VesselType, Status, Length, Width,
Draft, Cargo, TransceiverClass
"""
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

USECOLS = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading", "VesselType"]


def load_ais_window(csv_path, bbox, start_time, end_time, chunksize=500_000):
    """
    bbox: (min_lon, min_lat, max_lon, max_lat)
    start_time/end_time: pandas-parseable timestamps (UTC)
    Returns a single filtered DataFrame -- safe to call on a multi-GB monthly
    file because filtering happens per-chunk before concatenation, so peak
    memory stays bounded by chunksize, not file size.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    start_time, end_time = pd.Timestamp(start_time), pd.Timestamp(end_time)
    chunks = []
    for chunk in pd.read_csv(csv_path, usecols=USECOLS, parse_dates=["BaseDateTime"], chunksize=chunksize):
        m = (
            chunk["LON"].between(min_lon, max_lon)
            & chunk["LAT"].between(min_lat, max_lat)
            & chunk["BaseDateTime"].between(start_time, end_time)
        )
        if m.any():
            chunks.append(chunk.loc[m])
    if not chunks:
        return pd.DataFrame(columns=USECOLS)
    return pd.concat(chunks, ignore_index=True)


def filter_to_bilge_relevant_types(df):
    """IMO-reporting-relevant vessel types per the synopsis: tankers, bulk
    carriers/cargo, container ships, fishing vessels. AIS VesselType codes:
    70-79 cargo, 80-89 tanker, 30-39 fishing. Adjust the set if you also
    want passenger/other categories -- these three groups are the synopsis's
    explicit scope, not an AIS-spec requirement."""
    relevant = set(range(70, 90)) | set(range(30, 40))
    return df[df["VesselType"].isin(relevant)].copy()


def to_geodataframe(df, src_crs="EPSG:4326"):
    """Attach point geometry in WGS84. Reproject downstream with
    geo_utils.crs_utils.reproject_points_to() once you know the SAR scene's
    UTM zone -- don't hardcode a UTM zone in this module."""
    geom = [Point(xy) for xy in zip(df["LON"], df["LAT"])]
    return gpd.GeoDataFrame(df, geometry=geom, crs=src_crs)


if __name__ == "__main__":
    df = load_ais_window(
        "AIS_2026_03_Zone15.csv",
        bbox=(-94.0, 27.0, -88.0, 30.0),
        start_time="2026-03-15 00:00",
        end_time="2026-03-15 12:00",
    )
    print(f"{len(df)} AIS reports in window")
    gdf = to_geodataframe(filter_to_bilge_relevant_types(df))
    print(gdf.head())
