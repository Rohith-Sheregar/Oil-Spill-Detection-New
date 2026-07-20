"""
Forcing data for the drift model: ERA5 wind (atmosphere) via cdsapi, CMEMS
ocean currents via the copernicusmarine toolbox. Both become OpenDrift
readers -- see src/drift/lagrangian_drift.py.

ERA5 (post-2025 CDS migration): ~/.cdsapirc must contain
    url: https://cds.climate.copernicus.eu/api
    key: <personal access token from https://cds.climate.copernicus.eu>
If you have an old .cdsapirc pointing at the pre-2025 URL, requests fail
with an auth error that looks unrelated to the URL itself -- regenerate the
file rather than debugging the credential.

CMEMS: `pip install copernicusmarine`, then `copernicusmarine login` once
(interactive) before this will work non-interactively.
"""
import cdsapi
import copernicusmarine


def fetch_era5_wind(area, date, out_path="era5_wind.nc"):
    """
    area: (N, W, S, E) in degrees -- NOTE this order, different from the
          (min_lon, min_lat, max_lon, max_lat) bbox convention used for AIS
          and Sentinel-1 elsewhere in this project. Easy to transpose by
          habit -- that's a silent-wrong-area bug, not a crash, so double
          check this specific call site.
    date: 'YYYY-MM-DD'
    """
    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
            "year": [date[:4]], "month": [date[5:7]], "day": [date[8:10]],
            "time": [f"{h:02d}:00" for h in range(0, 24, 3)],
            "area": list(area),
            "data_format": "netcdf",
        },
        out_path,
    )
    return out_path


def fetch_cmems_currents(bbox, start_date, end_date, out_path="cmems_currents.nc"):
    """
    bbox: (min_lon, max_lon, min_lat, max_lat)
    dataset_id below is the standard global physics analysis/forecast
    product (1/12 deg) referenced in the synopsis. CMEMS product ids do get
    renamed/superseded over time -- if this 404s, search the current id at
    data.marine.copernicus.eu rather than assuming the call signature itself
    is wrong.
    """
    min_lon, max_lon, min_lat, max_lat = bbox
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_PT1H-m",
        variables=["uo", "vo"],
        minimum_longitude=min_lon, maximum_longitude=max_lon,
        minimum_latitude=min_lat, maximum_latitude=max_lat,
        start_datetime=start_date, end_datetime=end_date,
        output_filename=out_path,
    )
    return out_path


if __name__ == "__main__":
    fetch_era5_wind(area=(30, -94, 27, -88), date="2026-03-15")
    fetch_cmems_currents(bbox=(-94, -88, 27, 30), start_date="2026-03-15", end_date="2026-03-16")
