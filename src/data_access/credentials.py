"""
Credential loader for Copernicus portals.

Reads outputs/.env (relative to the project root) and injects variables into
os.environ so that sentinel1_cdse.py, era5_cmems.py, and copernicusmarine can
all pick them up transparently.

Usage (at the top of any script or notebook):
    from src.data_access.credentials import load_env
    load_env()

The .env file is git-ignored (outputs/ is in .gitignore).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# Project root = two levels up from this file (src/data_access/credentials.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH     = _PROJECT_ROOT / "outputs" / ".env"


def load_env(env_path: str | Path | None = None, override: bool = False) -> dict[str, str]:
    """
    Parse a .env file and inject key=value pairs into os.environ.

    Parameters
    ----------
    env_path : optional path to .env file; defaults to outputs/.env
    override : if True, overwrite existing env vars; default False (safe mode)

    Returns
    -------
    dict of the variables loaded (for inspection / logging)
    """
    path = Path(env_path) if env_path else _ENV_PATH

    if not path.exists():
        log.warning("Credentials file not found: %s — skipping.", path)
        return {}

    loaded: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            # Skip blank lines and comments
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                if override or key not in os.environ:
                    os.environ[key] = value
                    loaded[key] = value

    if loaded:
        log.info("Loaded %d credential(s) from %s", len(loaded), path)
    return loaded


def get_cdse_token() -> str:
    """Convenience: load env then return a fresh CDSE access token."""
    load_env()
    from src.data_access.sentinel1_cdse import get_access_token
    return get_access_token(
        username=os.environ["CDSE_USER"],
        password=os.environ["CDSE_PASS"],
    )
