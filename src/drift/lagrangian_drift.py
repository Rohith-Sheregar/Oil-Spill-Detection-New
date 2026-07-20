"""
Bidirectional Lagrangian drift attribution via OpenDrift/OpenOil -- don't
hand-roll the particle-drift physics. OpenOil already implements wind- and
current-forced drift with a NOAA ADIOS/PyGnome-derived weathering model
(evaporation, emulsification, entrainment). Note: this is NOT literally the
Fay (1971) spreading law some cited literature names -- arguably more
complete, but call it out as a deliberate substitution in any writeup that
references Fay specifically.

Forward run: seed at a Tier-1 candidate vessel's position at the estimated
discharge time, run forward to the SAR acquisition time.
Backward run: seed across the detected spill polygon at the SAR acquisition
time, run with a NEGATIVE time step to trace backward toward a probable
origin.

API caveat (read before debugging silently): `o.history['lon']`/`['lat']`
below are OpenDrift's internal per-element position history, stored as
masked arrays shaped (elements, timesteps). This has been a stable internal
representation across recent releases, but isn't part of the documented
public API, and the negative-time_step backward-run pattern should be
double-checked against your installed version's tutorial/changelog before
you trust the output -- this is the single most likely piece of this
scaffold to need a one-line fix for your exact OpenDrift version.
"""
from datetime import timedelta
import numpy as np
from opendrift.models.openoil import OpenOil
from opendrift.readers import reader_netCDF_CF_generic


def _build_model(wind_nc, current_nc, loglevel=20):
    o = OpenOil(loglevel=loglevel)
    o.add_reader(reader_netCDF_CF_generic.Reader(wind_nc))
    o.add_reader(reader_netCDF_CF_generic.Reader(current_nc))
    return o


def _final_positions(o):
    """Last unmasked timestep's (lon, lat) per element. See module
    docstring caveat on o.history -- confirm field names if this errors."""
    lon, lat = o.history["lon"], o.history["lat"]
    lon_final = lon[:, -1].compressed() if hasattr(lon, "compressed") else np.asarray(lon[:, -1])
    lat_final = lat[:, -1].compressed() if hasattr(lat, "compressed") else np.asarray(lat[:, -1])
    return lon_final, lat_final


def run_forward_from_vessel(vessel_lon, vessel_lat, discharge_time, wind_nc, current_nc,
                             duration_hours=6, number=1000, seed_radius_m=500):
    o = _build_model(wind_nc, current_nc)
    o.seed_elements(lon=vessel_lon, lat=vessel_lat, time=discharge_time,
                     number=number, radius=seed_radius_m)
    o.run(duration=timedelta(hours=duration_hours), time_step=900)
    return _final_positions(o)


def run_backward_from_spill(spill_lons, spill_lats, detection_time, wind_nc, current_nc,
                             duration_hours=6, time_step=900):
    o = _build_model(wind_nc, current_nc)
    o.seed_elements(lon=np.asarray(spill_lons), lat=np.asarray(spill_lats), time=detection_time)
    o.run(duration=timedelta(hours=duration_hours), time_step=-time_step)  # negative = backward
    return _final_positions(o)


def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def bidirectional_match_score(forward_lonlat, backward_lonlat, decay_km=20.0):
    """
    Centroid-distance similarity mapped to [0,1] via exponential decay
    (1.0 = perfect overlap, ->0 as distance grows). decay_km is a tuning
    knob, not a physical constant -- ~20km is a reasonable start (a few SAR
    pixels of slack plus typical drift-model positional uncertainty over a
    few hours), but sanity-check it against your confirmed incidents rather
    than trusting the default.
    """
    fwd_lon, fwd_lat = np.mean(forward_lonlat[0]), np.mean(forward_lonlat[1])
    bwd_lon, bwd_lat = np.mean(backward_lonlat[0]), np.mean(backward_lonlat[1])
    dist_km = haversine_km(fwd_lon, fwd_lat, bwd_lon, bwd_lat)
    return float(np.exp(-dist_km / decay_km))


if __name__ == "__main__":
    fwd = run_forward_from_vessel(
        vessel_lon=-90.2, vessel_lat=28.1, discharge_time="2026-03-15 06:00",
        wind_nc="era5_wind.nc", current_nc="cmems_currents.nc",
    )
    bwd = run_backward_from_spill(
        spill_lons=[-90.15, -90.16, -90.17], spill_lats=[28.15, 28.16, 28.17],
        detection_time="2026-03-15 12:00",
        wind_nc="era5_wind.nc", current_nc="cmems_currents.nc",
    )
    print("S_drift =", bidirectional_match_score(fwd, bwd))
