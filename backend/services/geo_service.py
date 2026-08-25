"""
geo_service.py

Member 1 — Satellite Data & Geospatial

Wraps the SAR image geolocation logic (bounding box, timestamp, pixel <->
lat/lon conversion, mask -> polygon conversion) as importable functions,
matching the calling convention used elsewhere in backend/services/
(see ml_service.detect_spill, drift_service.estimate_origin).

Handles both cases seen in real Sentinel-1 data:
  - Images with a proper CRS + affine transform
  - Raw Sentinel-1 GRD images that only carry Ground Control Points (GCPs),
    no CRS/affine transform — this is what real downloaded Sentinel-1
    scenes look like out of the box.
"""

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features
from rasterio.warp import transform_bounds, transform_geom
from shapely.geometry import shape, mapping


# ---------------------------------------------------------------------
# Timestamp parsing (filename-based, for real Sentinel-1 naming convention)
# ---------------------------------------------------------------------

_MEASUREMENT_TIFF_RE = re.compile(
    r"s1[ab]-[a-z0-9]+-[a-z]+-[a-z]+-"
    r"(?P<start>\d{8}t\d{6})-"
    r"(?P<stop>\d{8}t\d{6})-",
    re.IGNORECASE,
)

_SAFE_PRODUCT_RE = re.compile(
    r"S1[AB]_[A-Z0-9]+_[A-Z0-9]+_[A-Z0-9]+_"
    r"(?P<start>\d{8}T\d{6})_"
    r"(?P<stop>\d{8}T\d{6})_",
)


def _parse_timestamp_from_filename(filename: str):
    name = Path(filename).name
    match = _MEASUREMENT_TIFF_RE.search(name) or _SAFE_PRODUCT_RE.search(name)
    if not match:
        return None
    fmt = "%Y%m%dt%H%M%S"
    start = datetime.strptime(match.group("start").lower(), fmt)
    stop = datetime.strptime(match.group("stop").lower(), fmt)
    return {"start": start.isoformat(), "stop": stop.isoformat()}


# ---------------------------------------------------------------------
# Core metadata extraction — the geospatial equivalent of detect_spill()
# ---------------------------------------------------------------------

def extract_image_metadata(image_path: str) -> dict:
    """
    Extracts bounding box, resolution, and acquisition time from a
    georeferenced SAR/optical image.

    Mirrors the calling convention of ml_service.detect_spill(image_path):
    takes a path, returns a plain dict.
    """
    with rasterio.open(image_path) as src:
        width, height = src.width, src.height
        crs = src.crs
        transform = src.transform
        gcps, gcp_crs = src.get_gcps() if src.gcps else (None, None)
        using_gcps = crs is None and gcps

        if using_gcps:
            gcp_lons = [gcp.x for gcp in gcps]
            gcp_lats = [gcp.y for gcp in gcps]
            lon_min, lon_max = min(gcp_lons), max(gcp_lons)
            lat_min, lat_max = min(gcp_lats), max(gcp_lats)
        elif crs is not None:
            lon_min, lat_min, lon_max, lat_max = transform_bounds(
                crs, "EPSG:4326", *src.bounds
            )
        else:
            lon_min = lat_min = lon_max = lat_max = None

        tags = src.tags()
        acquisition_time = (
            tags.get("TIFFTAG_DATETIME")
            or tags.get("ACQUISITION_DATETIME")
            or tags.get("DATETIME")
        )
        if acquisition_time is None:
            parsed = _parse_timestamp_from_filename(image_path)
            acquisition_time = parsed if parsed else None

    return {
        "width": width,
        "height": height,
        "has_crs": crs is not None,
        "crs": str(crs) if crs else None,
        "bounding_box": {
            "lon_min": lon_min,
            "lat_min": lat_min,
            "lon_max": lon_max,
            "lat_max": lat_max,
        },
        "acquisition_time": acquisition_time,
    }


# ---------------------------------------------------------------------
# Pixel -> lat/lon conversion (single point)
# ---------------------------------------------------------------------

def _pixel_to_latlon_gcp(row: float, col: float, gcps) -> tuple[float, float]:
    weights, lats, lons = [], [], []
    for gcp in gcps:
        dist = ((gcp.row - row) ** 2 + (gcp.col - col) ** 2) ** 0.5
        if dist < 1e-6:
            return gcp.y, gcp.x
        w = 1.0 / dist
        weights.append(w)
        lats.append(gcp.y * w)
        lons.append(gcp.x * w)
    total = sum(weights)
    return sum(lats) / total, sum(lons) / total


def pixel_to_latlon(row: float, col: float, image_path: str) -> tuple[float, float]:
    """
    Converts a single pixel (row, col) to (lat, lon).
    Auto-detects whether the image has a CRS/affine transform or only
    GCPs, and routes accordingly.
    """
    with rasterio.open(image_path) as src:
        if src.crs is not None:
            x, y = src.transform * (col, row)
            if src.crs.to_string() != "EPSG:4326":
                lon, lat = transform_bounds(src.crs, "EPSG:4326", x, y, x, y)[:2]
            else:
                lon, lat = x, y
            return lat, lon
        else:
            gcps, _ = src.get_gcps()
            if not gcps:
                raise ValueError(f"{image_path}: no CRS and no GCPs — cannot geolocate.")
            return _pixel_to_latlon_gcp(row, col, gcps)


# ---------------------------------------------------------------------
# Mask -> geographic polygon conversion
# ---------------------------------------------------------------------

def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.max() > 1:
        mask = (mask > 127).astype("uint8")
    else:
        mask = (mask > 0).astype("uint8")
    return mask


def mask_to_polygons(mask: np.ndarray, image_path: str, min_area_px: int = 10) -> list[dict]:
    """
    Converts a binary spill mask (same pixel dimensions as the source
    image) into geographic polygons.

    This is what Member 3 (drift) actually needs as input — not just a
    single point, but a mapped-out spill boundary.

    Expects: mask as a 2D numpy array (0/1 or 0/255), same (height, width)
    as the image at image_path. If ml_service.detect_spill() is updated
    to return a real mask array (it currently doesn't — see note below),
    call this as: mask_to_polygons(spill_result["mask"], image_path)

    Returns: list of dicts, one per disconnected spill region:
        {
          "polygon_latlon": [[lat, lon], ...],
          "centroid_latlon": [lat, lon],
          "area_px": int
        }
    """
    mask = _normalize_mask(mask)

    with rasterio.open(image_path) as src:
        crs = src.crs
        transform = src.transform
        gcps, _ = src.get_gcps() if src.gcps else (None, None)

    results = []
    identity = rasterio.Affine.identity()
    use_transform = transform if crs is not None else identity

    for geom, value in features.shapes(mask, mask=mask.astype(bool), transform=use_transform):
        if value != 1:
            continue
        poly = shape(geom)
        if poly.area < min_area_px:
            continue

        if crs is not None:
            if crs.to_string() != "EPSG:4326":
                geom_wgs84 = transform_geom(crs, "EPSG:4326", mapping(poly))
                poly = shape(geom_wgs84)
            coords_latlon = [[lat, lon] for lon, lat in poly.exterior.coords]
            centroid_latlon = [poly.centroid.y, poly.centroid.x]
        else:
            if not gcps:
                raise ValueError(f"{image_path}: no CRS and no GCPs — cannot geolocate mask.")
            coords_latlon = [
                list(_pixel_to_latlon_gcp(row, col, gcps))
                for col, row in poly.exterior.coords
            ]
            centroid_latlon = list(
                _pixel_to_latlon_gcp(poly.centroid.y, poly.centroid.x, gcps)
            )

        results.append({
            "polygon_latlon": coords_latlon,
            "centroid_latlon": centroid_latlon,
            "area_px": int(np.sum(mask)),
        })

    return results
