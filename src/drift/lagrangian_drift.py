"""Bidirectional Lagrangian drift attribution for oil-slick forensics.

The public functions in this module prefer an OpenDrift/OpenOil simulation
when OpenDrift is installed. If that dependency or its readers are unavailable,
the module falls back to a small 2D advection-diffusion solver driven by NetCDF
wind and surface-current fields.

The fallback solver treats wave Stokes drift as 1.5 percent of the 10 m wind
velocity and adds it to the surface current velocity before integrating particle
positions. This is intentionally conservative: it preserves the Module 4 API in
lightweight environments while keeping the forcing terms explicit and auditable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np

log = logging.getLogger(__name__)

EARTH_RADIUS_M: float = 6_371_000.0
STOKES_WIND_FRACTION: float = 0.015
DEFAULT_TIME_STEP_SECONDS: int = 900
DEFAULT_DIFFUSIVITY_M2_S: float = 2.0
DEFAULT_SEED_RADIUS_M: float = 500.0
DEFAULT_BACKWARD_SEED_RADIUS_M: float = 750.0

_WIND_U_ALIASES: tuple[str, ...] = (
    "u",
    "u10",
    "10u",
    "eastward_wind",
    "eastward_wind_velocity",
    "x_wind",
)
_WIND_V_ALIASES: tuple[str, ...] = (
    "v",
    "v10",
    "10v",
    "northward_wind",
    "northward_wind_velocity",
    "y_wind",
)
_CURRENT_U_ALIASES: tuple[str, ...] = (
    "uo",
    "eastward_sea_water_velocity",
    "surface_eastward_sea_water_velocity",
)
_CURRENT_V_ALIASES: tuple[str, ...] = (
    "vo",
    "northward_sea_water_velocity",
    "surface_northward_sea_water_velocity",
)


class _VectorField(Protocol):
    """Minimal velocity-field interface used by the fallback solver."""

    def sample(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        when: datetime,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return eastward and northward velocities in metres per second."""


@dataclass
class _ZeroVectorField:
    """Velocity field used when no readable NetCDF forcing is available."""

    name: str

    def sample(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        when: datetime,
    ) -> tuple[np.ndarray, np.ndarray]:
        del when
        return np.zeros_like(lon, dtype=np.float64), np.zeros_like(lat, dtype=np.float64)


class _XarrayVectorField:
    """Nearest-neighbour sampler around an xarray-backed vector field."""

    def __init__(
        self,
        nc_path: str | Path,
        u_aliases: Iterable[str],
        v_aliases: Iterable[str],
        name: str,
    ) -> None:
        try:
            import xarray as xr
        except Exception as exc:  # pragma: no cover - depends on optional env
            raise ImportError("xarray is required for fallback NetCDF sampling") from exc

        self.path = Path(nc_path)
        self.name = name
        self._ds = xr.open_dataset(self.path)
        self._u = self._prepare_dataarray(self._select_var(u_aliases))
        self._v = self._prepare_dataarray(self._select_var(v_aliases))
        self._lon_name = self._find_coord_name(("longitude", "lon", "x"))
        self._lat_name = self._find_coord_name(("latitude", "lat", "y"))
        self._time_name = self._find_coord_name(("time", "valid_time"), required=False)

    def _select_var(self, aliases: Iterable[str]) -> Any:
        lower_to_actual = {name.lower(): name for name in self._ds.data_vars}
        for alias in aliases:
            actual = lower_to_actual.get(alias.lower())
            if actual is not None:
                return self._ds[actual]
        raise KeyError(
            f"{self.name} NetCDF {self.path} does not contain any of {tuple(aliases)}. "
            f"Available variables: {tuple(self._ds.data_vars)}"
        )

    def _find_coord_name(
        self,
        candidates: tuple[str, ...],
        required: bool = True,
    ) -> str | None:
        names = set(self._u.coords) | set(self._u.dims)
        lower_to_actual = {name.lower(): name for name in names}
        for candidate in candidates:
            actual = lower_to_actual.get(candidate.lower())
            if actual is not None:
                return actual
        if required:
            raise KeyError(
                f"{self.name} NetCDF {self.path} lacks coordinate {candidates}; "
                f"available dims={self._u.dims}, coords={tuple(self._u.coords)}"
            )
        return None

    def _prepare_dataarray(self, arr: Any) -> Any:
        keep_names = {"time", "valid_time", "latitude", "lat", "longitude", "lon", "x", "y"}
        selectors: dict[str, int] = {}
        for dim in arr.dims:
            if dim.lower() not in keep_names:
                selectors[dim] = 0
        if selectors:
            arr = arr.isel(selectors)
        return arr.squeeze(drop=True)

    def _normalise_lon(self, lon: np.ndarray) -> np.ndarray:
        if self._lon_name is None:
            return lon
        coord = np.asarray(self._u[self._lon_name].values, dtype=np.float64)
        if coord.size == 0:
            return lon
        if np.nanmin(coord) >= 0.0 and np.nanmax(coord) > 180.0:
            return np.mod(lon, 360.0)
        return ((lon + 180.0) % 360.0) - 180.0

    @staticmethod
    def _to_naive_utc(when: datetime) -> np.datetime64:
        if when.tzinfo is not None:
            when = when.astimezone(timezone.utc).replace(tzinfo=None)
        return np.datetime64(when)

    def _sample_one(self, arr: Any, lon: float, lat: float, when: datetime) -> float:
        selectors: dict[str, Any] = {
            self._lon_name: lon,
            self._lat_name: lat,
        }
        if self._time_name is not None and self._time_name in arr.dims:
            selectors[self._time_name] = self._to_naive_utc(when)
        try:
            value = arr.sel(selectors, method="nearest").values
            return float(np.asarray(value).squeeze())
        except Exception as exc:
            log.debug("%s sample failed at lon=%.4f lat=%.4f: %s", self.name, lon, lat, exc)
            return 0.0

    def sample(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        when: datetime,
    ) -> tuple[np.ndarray, np.ndarray]:
        lon_arr = self._normalise_lon(np.asarray(lon, dtype=np.float64))
        lat_arr = np.asarray(lat, dtype=np.float64)
        u = np.empty_like(lon_arr, dtype=np.float64)
        v = np.empty_like(lat_arr, dtype=np.float64)
        for idx, (lo, la) in enumerate(zip(lon_arr, lat_arr)):
            u[idx] = self._sample_one(self._u, float(lo), float(la), when)
            v[idx] = self._sample_one(self._v, float(lo), float(la), when)
        return np.nan_to_num(u, nan=0.0), np.nan_to_num(v, nan=0.0)


def _coerce_datetime(value: str | datetime | np.datetime64) -> datetime:
    """Parse common timestamp inputs into a timezone-aware UTC datetime."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, np.datetime64):
        seconds = value.astype("datetime64[s]").astype(int)
        dt = datetime.fromtimestamp(int(seconds), tz=timezone.utc)
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            # Last-resort parser for strings like "2026-03-15 06:00".
            dt = datetime.strptime(text, "%Y-%m-%d %H:%M")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _validate_lon_lat(lon: float, lat: float, label: str) -> tuple[float, float]:
    lon_f = float(lon)
    lat_f = float(lat)
    if not np.isfinite(lon_f) or not np.isfinite(lat_f):
        raise ValueError(f"{label} coordinates must be finite; got lon={lon}, lat={lat}.")
    if not -180.0 <= lon_f <= 360.0:
        raise ValueError(f"{label} longitude out of range: {lon_f}.")
    if not -90.0 <= lat_f <= 90.0:
        raise ValueError(f"{label} latitude out of range: {lat_f}.")
    if lon_f > 180.0:
        lon_f = ((lon_f + 180.0) % 360.0) - 180.0
    return lon_f, lat_f


def _validate_particle_count(num_particles: int) -> int:
    n = int(num_particles)
    if n <= 0:
        raise ValueError(f"num_particles must be positive; got {num_particles}.")
    return n


def _validate_duration(duration_hours: float) -> float:
    duration = float(duration_hours)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"duration_hours must be a positive finite number; got {duration_hours}.")
    return duration


def _safe_reader(path: str | Path | None, u_aliases: Iterable[str], v_aliases: Iterable[str], name: str) -> _VectorField:
    if path is None:
        log.warning("%s forcing not provided; using zero %s velocity.", name, name.lower())
        return _ZeroVectorField(name=name)
    nc_path = Path(path)
    if not nc_path.exists():
        log.warning("%s forcing file not found at %s; using zero velocity.", name, nc_path)
        return _ZeroVectorField(name=name)
    try:
        return _XarrayVectorField(nc_path, u_aliases, v_aliases, name=name)
    except Exception as exc:
        log.warning("%s forcing could not be read from %s: %s. Using zero velocity.", name, nc_path, exc)
        return _ZeroVectorField(name=name)


def _seed_disc(
    center_lon: float,
    center_lat: float,
    number: int,
    radius_m: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Seed particles uniformly inside a disc around a WGS84 point."""
    theta = rng.uniform(0.0, 2.0 * np.pi, number)
    radius = radius_m * np.sqrt(rng.uniform(0.0, 1.0, number))
    east_m = radius * np.cos(theta)
    north_m = radius * np.sin(theta)
    return _advance_lon_lat(
        np.full(number, center_lon, dtype=np.float64),
        np.full(number, center_lat, dtype=np.float64),
        east_m,
        north_m,
    )


def _advance_lon_lat(
    lon: np.ndarray,
    lat: np.ndarray,
    east_m: np.ndarray,
    north_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Move lon/lat points by local east/north metre displacements."""
    lat_rad = np.radians(lat)
    dlat = (north_m / EARTH_RADIUS_M) * (180.0 / np.pi)
    denom = EARTH_RADIUS_M * np.cos(lat_rad)
    denom = np.where(np.abs(denom) < 1.0, 1.0, denom)
    dlon = (east_m / denom) * (180.0 / np.pi)
    new_lat = np.clip(lat + dlat, -89.999, 89.999)
    new_lon = ((lon + dlon + 180.0) % 360.0) - 180.0
    return new_lon.astype(np.float64), new_lat.astype(np.float64)


def _run_opendrift(
    lon: float,
    lat: float,
    start_time: datetime,
    wind_nc: str | Path | None,
    current_nc: str | Path | None,
    duration_hours: float,
    num_particles: int,
    seed_radius_m: float,
    backward: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run OpenDrift if importable; return None when unavailable or failed."""
    if wind_nc is None or current_nc is None:
        return None
    try:
        from opendrift.models.openoil import OpenOil
        from opendrift.readers import reader_netCDF_CF_generic
    except Exception as exc:  # pragma: no cover - optional dependency
        log.info("OpenDrift unavailable, using fallback drift solver: %s", exc)
        return None

    try:
        model = OpenOil(loglevel=30)
        model.add_reader(reader_netCDF_CF_generic.Reader(str(wind_nc)))
        model.add_reader(reader_netCDF_CF_generic.Reader(str(current_nc)))
        for key in (
            "drift:wind_drift_factor",
            "drift:stokes_drift_factor",
            "processes:stokes_drift",
        ):
            try:
                if key.endswith("factor"):
                    model.set_config(key, STOKES_WIND_FRACTION)
                else:
                    model.set_config(key, True)
            except Exception:
                continue
        model.seed_elements(
            lon=lon,
            lat=lat,
            time=start_time,
            number=num_particles,
            radius=seed_radius_m,
        )
        step = -DEFAULT_TIME_STEP_SECONDS if backward else DEFAULT_TIME_STEP_SECONDS
        model.run(duration=timedelta(hours=duration_hours), time_step=step)
        return _extract_opendrift_final_positions(model)
    except Exception as exc:
        log.warning("OpenDrift simulation failed; using fallback solver: %s", exc)
        return None


def _extract_opendrift_final_positions(model: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract final per-particle positions from an OpenDrift model history."""
    lon_hist = model.history["lon"]
    lat_hist = model.history["lat"]

    lon_arr = np.ma.asarray(lon_hist)
    lat_arr = np.ma.asarray(lat_hist)
    final_lon: list[float] = []
    final_lat: list[float] = []

    for idx in range(lon_arr.shape[0]):
        valid = ~(np.ma.getmaskarray(lon_arr[idx]) | np.ma.getmaskarray(lat_arr[idx]))
        if not np.any(valid):
            continue
        last = np.flatnonzero(valid)[-1]
        final_lon.append(float(lon_arr[idx, last]))
        final_lat.append(float(lat_arr[idx, last]))

    if not final_lon:
        raise RuntimeError("OpenDrift returned no valid final particle positions.")
    return np.asarray(final_lon, dtype=np.float64), np.asarray(final_lat, dtype=np.float64)


def _run_fallback_solver(
    lon: float,
    lat: float,
    start_time: datetime,
    wind_nc: str | Path | None,
    current_nc: str | Path | None,
    duration_hours: float,
    num_particles: int,
    seed_radius_m: float,
    backward: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a surface advection-diffusion particle cloud."""
    wind = _safe_reader(wind_nc, _WIND_U_ALIASES, _WIND_V_ALIASES, "Wind")
    current = _safe_reader(current_nc, _CURRENT_U_ALIASES, _CURRENT_V_ALIASES, "Current")

    rng_seed = 4_204 if backward else 4_203
    rng = np.random.default_rng(rng_seed)
    lons, lats = _seed_disc(lon, lat, num_particles, seed_radius_m, rng)

    sign = -1.0 if backward else 1.0
    total_seconds = duration_hours * 3600.0
    n_steps = max(1, int(np.ceil(total_seconds / DEFAULT_TIME_STEP_SECONDS)))
    dt_abs = total_seconds / n_steps
    diffusion_sigma_m = float(np.sqrt(2.0 * DEFAULT_DIFFUSIVITY_M2_S * dt_abs))

    for step_idx in range(n_steps):
        elapsed = timedelta(seconds=sign * step_idx * dt_abs)
        when = start_time + elapsed
        wind_u, wind_v = wind.sample(lons, lats, when)
        cur_u, cur_v = current.sample(lons, lats, when)

        east_velocity = cur_u + STOKES_WIND_FRACTION * wind_u
        north_velocity = cur_v + STOKES_WIND_FRACTION * wind_v

        east_m = sign * east_velocity * dt_abs + rng.normal(0.0, diffusion_sigma_m, num_particles)
        north_m = sign * north_velocity * dt_abs + rng.normal(0.0, diffusion_sigma_m, num_particles)
        lons, lats = _advance_lon_lat(lons, lats, east_m, north_m)

    valid = np.isfinite(lons) & np.isfinite(lats)
    if not np.any(valid):
        raise RuntimeError("Fallback drift solver produced no finite particle positions.")
    return lons[valid], lats[valid]


def run_forward_simulation(
    vessel_lon: float,
    vessel_lat: float,
    discharge_time: str | datetime | np.datetime64,
    wind_nc: str | Path | None,
    current_nc: str | Path | None,
    duration_hours: float = 6.0,
    num_particles: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate oil particles forward from a candidate vessel position.

    Args:
        vessel_lon: Vessel longitude in WGS84 degrees.
        vessel_lat: Vessel latitude in WGS84 degrees.
        discharge_time: Estimated discharge timestamp. Naive timestamps are
            interpreted as UTC.
        wind_nc: ERA5 10 m wind NetCDF containing east/north wind components.
        current_nc: CMEMS surface-current NetCDF containing east/north currents.
        duration_hours: Simulation duration from discharge to SAR acquisition.
        num_particles: Number of particles to seed around the vessel.

    Returns:
        Tuple ``(lons, lats)`` of final particle coordinates in WGS84 degrees.

    Raises:
        ValueError: If coordinates, duration, or particle count are invalid.
        RuntimeError: If both OpenDrift and fallback simulation fail.
    """
    lon, lat = _validate_lon_lat(vessel_lon, vessel_lat, "vessel")
    duration = _validate_duration(duration_hours)
    n_particles = _validate_particle_count(num_particles)
    start_time = _coerce_datetime(discharge_time)

    result = _run_opendrift(
        lon=lon,
        lat=lat,
        start_time=start_time,
        wind_nc=wind_nc,
        current_nc=current_nc,
        duration_hours=duration,
        num_particles=n_particles,
        seed_radius_m=DEFAULT_SEED_RADIUS_M,
        backward=False,
    )
    if result is not None:
        return result

    return _run_fallback_solver(
        lon=lon,
        lat=lat,
        start_time=start_time,
        wind_nc=wind_nc,
        current_nc=current_nc,
        duration_hours=duration,
        num_particles=n_particles,
        seed_radius_m=DEFAULT_SEED_RADIUS_M,
        backward=False,
    )


def run_backward_simulation(
    spill_centroid_lon: float,
    spill_centroid_lat: float,
    sar_time: str | datetime | np.datetime64,
    wind_nc: str | Path | None,
    current_nc: str | Path | None,
    duration_hours: float = 6.0,
    num_particles: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace a verified slick patch backward from SAR acquisition time.

    The function signature receives a slick centroid, so particles are seeded in
    a compact disc around that centroid as a proxy for the verified Module 2
    patch footprint. OpenDrift is used when available; otherwise the fallback
    solver integrates the same forcing terms with a negative advection step.

    Args:
        spill_centroid_lon: Slick centroid longitude in WGS84 degrees.
        spill_centroid_lat: Slick centroid latitude in WGS84 degrees.
        sar_time: SAR acquisition timestamp. Naive timestamps are interpreted as
            UTC.
        wind_nc: ERA5 10 m wind NetCDF containing east/north wind components.
        current_nc: CMEMS surface-current NetCDF containing east/north currents.
        duration_hours: Backward simulation duration.
        num_particles: Number of particles to seed around the slick centroid.

    Returns:
        Tuple ``(lons, lats)`` of predicted origin particle coordinates.
    """
    lon, lat = _validate_lon_lat(spill_centroid_lon, spill_centroid_lat, "spill")
    duration = _validate_duration(duration_hours)
    n_particles = _validate_particle_count(num_particles)
    start_time = _coerce_datetime(sar_time)

    result = _run_opendrift(
        lon=lon,
        lat=lat,
        start_time=start_time,
        wind_nc=wind_nc,
        current_nc=current_nc,
        duration_hours=duration,
        num_particles=n_particles,
        seed_radius_m=DEFAULT_BACKWARD_SEED_RADIUS_M,
        backward=True,
    )
    if result is not None:
        return result

    return _run_fallback_solver(
        lon=lon,
        lat=lat,
        start_time=start_time,
        wind_nc=wind_nc,
        current_nc=current_nc,
        duration_hours=duration,
        num_particles=n_particles,
        seed_radius_m=DEFAULT_BACKWARD_SEED_RADIUS_M,
        backward=True,
    )


def _particle_centroid(particles: tuple[np.ndarray, np.ndarray] | np.ndarray) -> tuple[float, float]:
    """Return the finite lon/lat centroid of a particle cloud."""
    if isinstance(particles, tuple):
        lon, lat = particles
    else:
        arr = np.asarray(particles, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("Particle array must be a tuple(lons, lats) or an (N, 2) array.")
        lon, lat = arr[:, 0], arr[:, 1]
    lon_arr = np.asarray(lon, dtype=np.float64)
    lat_arr = np.asarray(lat, dtype=np.float64)
    valid = np.isfinite(lon_arr) & np.isfinite(lat_arr)
    if not np.any(valid):
        raise ValueError("Particle cloud contains no finite coordinates.")
    return float(np.mean(lon_arr[valid])), float(np.mean(lat_arr[valid]))


def haversine_km(
    lon1: float | np.ndarray,
    lat1: float | np.ndarray,
    lon2: float | np.ndarray,
    lat2: float | np.ndarray,
) -> float | np.ndarray:
    """Compute great-circle distance between WGS84 coordinates in kilometres."""
    lon1_r, lat1_r, lon2_r, lat2_r = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )
    distance = 2.0 * (EARTH_RADIUS_M / 1000.0) * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    if np.isscalar(distance):
        return float(distance)
    return distance


def compute_drift_similarity(
    forward_particles: tuple[np.ndarray, np.ndarray] | np.ndarray,
    backward_particles: tuple[np.ndarray, np.ndarray] | np.ndarray,
    decay_km: float = 20.0,
) -> float:
    """Score agreement between forward and backward drift centroids.

    Args:
        forward_particles: Final particle cloud from
            :func:`run_forward_simulation`, either ``(lons, lats)`` or ``(N, 2)``.
        backward_particles: Origin particle cloud from
            :func:`run_backward_simulation`, either ``(lons, lats)`` or ``(N, 2)``.
        decay_km: Exponential decay distance in kilometres.

    Returns:
        ``S_drift = exp(-distance_km / decay_km)`` clipped to ``[0.0, 1.0]``.
    """
    decay = float(decay_km)
    if not np.isfinite(decay) or decay <= 0.0:
        raise ValueError(f"decay_km must be a positive finite number; got {decay_km}.")

    fwd_lon, fwd_lat = _particle_centroid(forward_particles)
    bwd_lon, bwd_lat = _particle_centroid(backward_particles)
    distance_km = float(haversine_km(fwd_lon, fwd_lat, bwd_lon, bwd_lat))
    score = float(np.exp(-distance_km / decay))
    return float(np.clip(score, 0.0, 1.0))


# Backwards-compatible aliases for the earlier scaffold.
def run_forward_from_vessel(
    vessel_lon: float,
    vessel_lat: float,
    discharge_time: str | datetime | np.datetime64,
    wind_nc: str | Path | None,
    current_nc: str | Path | None,
    duration_hours: float = 6.0,
    number: int = 1000,
    seed_radius_m: float = DEFAULT_SEED_RADIUS_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper around :func:`run_forward_simulation`."""
    del seed_radius_m
    return run_forward_simulation(
        vessel_lon,
        vessel_lat,
        discharge_time,
        wind_nc,
        current_nc,
        duration_hours=duration_hours,
        num_particles=number,
    )


def run_backward_from_spill(
    spill_lons: Iterable[float],
    spill_lats: Iterable[float],
    detection_time: str | datetime | np.datetime64,
    wind_nc: str | Path | None,
    current_nc: str | Path | None,
    duration_hours: float = 6.0,
    time_step: int = DEFAULT_TIME_STEP_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper using the mean of legacy spill vertices."""
    del time_step
    lon_arr = np.asarray(list(spill_lons), dtype=np.float64)
    lat_arr = np.asarray(list(spill_lats), dtype=np.float64)
    if lon_arr.size == 0 or lat_arr.size == 0:
        raise ValueError("spill_lons and spill_lats must not be empty.")
    return run_backward_simulation(
        float(np.nanmean(lon_arr)),
        float(np.nanmean(lat_arr)),
        detection_time,
        wind_nc,
        current_nc,
        duration_hours=duration_hours,
        num_particles=max(len(lon_arr), 1),
    )


def bidirectional_match_score(
    forward_lonlat: tuple[np.ndarray, np.ndarray] | np.ndarray,
    backward_lonlat: tuple[np.ndarray, np.ndarray] | np.ndarray,
    decay_km: float = 20.0,
) -> float:
    """Compatibility wrapper around :func:`compute_drift_similarity`."""
    return compute_drift_similarity(forward_lonlat, backward_lonlat, decay_km=decay_km)


__all__ = [
    "run_forward_simulation",
    "run_backward_simulation",
    "compute_drift_similarity",
    "haversine_km",
    "run_forward_from_vessel",
    "run_backward_from_spill",
    "bidirectional_match_score",
]
