"""
Sentinel-1 GRD search + download via the Copernicus Data Space Ecosystem (CDSE).

CDSE replaced the old SciHub/DHuS system. `sentinelsat` is archived and will
NOT work against CDSE -- don't reach for it out of habit. This module uses
the documented OData + Keycloak token pattern.

Free account: https://dataspace.copernicus.eu

Note: the exact download host has shifted before in CDSE's history (catalogue
vs. a dedicated "zipper" subdomain) -- if `download_product` 404s, check
https://documentation.dataspace.copernicus.eu for the current value. The
token endpoint and OData filter/query pattern below are the stable parts.
"""
import os
import requests

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"


def get_access_token(username: str, password: str) -> str:
    """Keycloak token. Tokens expire (~10 min) -- call this again right
    before any long-running download rather than caching for a whole
    multi-hour session."""
    resp = requests.post(
        TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "username": username,
            "password": password,
            "grant_type": "password",
        },
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def search_sentinel1_grd(bbox, start_date, end_date, max_results=50):
    """
    bbox: (min_lon, min_lat, max_lon, max_lat)
    start_date/end_date: 'YYYY-MM-DD'
    Returns a list of product dicts (Id, Name, ContentDate, GeoFootprint, ...)
    filtered client-side to IW GRDH products.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    wkt = (
        f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},"
        f"{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"
    )
    filter_str = (
        "Collection/Name eq 'SENTINEL-1' and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') and "
        f"ContentDate/Start gt {start_date}T00:00:00.000Z and "
        f"ContentDate/Start lt {end_date}T00:00:00.000Z"
    )
    params = {"$filter": filter_str, "$top": max_results, "$orderby": "ContentDate/Start desc"}
    resp = requests.get(CATALOG_URL, params=params)
    resp.raise_for_status()
    products = resp.json()["value"]

    # OData attribute-filter syntax for productType varies by collection
    # version, so filter client-side on the product Name instead -- it's
    # easy to verify by eye, e.g. "S1A_IW_GRDH_1SDV_20260315T...".
    return [p for p in products if "IW_GRDH" in p["Name"]]


def download_product(product_id: str, out_path: str, token: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Streams the product zip to disk. Sentinel-1 IW GRD scenes are
    typically 800MB-1.6GB -- budget disk space and time accordingly, and
    don't load the response into memory."""
    headers = {"Authorization": f"Bearer {token}"}
    url = DOWNLOAD_URL.format(product_id=product_id)
    with requests.get(url, headers=headers, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    return out_path


if __name__ == "__main__":
    # Example: Gulf of Mexico, one-week window
    products = search_sentinel1_grd(
        bbox=(-94.0, 27.0, -88.0, 30.0),
        start_date="2026-03-01",
        end_date="2026-03-08",
    )
    print(f"Found {len(products)} IW GRDH scenes")
    if products:
        token = get_access_token(os.environ["CDSE_USER"], os.environ["CDSE_PASS"])
        download_product(products[0]["Id"], "scene_001.zip", token)
