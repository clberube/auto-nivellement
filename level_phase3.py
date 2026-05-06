from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from level_core import make_progress

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


# -------------------------- User-editable settings --------------------------
ELEMENT = "Ba"
ELEMENT_FILE_STEM = ELEMENT.lower()
INPUT_SHP = Path(f"output/{ELEMENT_FILE_STEM}_phase2_leveled_partial_overlap.shp")
OUTPUT_TIF = Path(f"output/{ELEMENT_FILE_STEM}_phase3_imp_ordinary_kriging.tif")

VALUE_COLUMN = f"{ELEMENT}_imp"
PHASE2_STATUS_COLUMN = "P2_STAT"
FILTER_EXCLUDED_PHASE2_POINTS = True
PHASE2_EXCLUDED_STATUS_PREFIXES = ("excluded_",)

# Grid settings (in target CRS units, meters if projected)
TARGET_CRS = None  # e.g. "EPSG:32198"; None => keep projected CRS or auto-UTM
PIXEL_SIZE = 1_000.0

# Ordinary kriging settings
# A log transform is usually gentler for positive geochemistry values.
LOG_TRANSFORM_VALUES = True
KRIGING_K_NEIGHBORS = 24
KRIGING_MIN_NEIGHBORS = 0
KRIGING_MAX_DISTANCE = 5_000.0  # meters; set <=0 for unlimited
QUERY_CHUNK_SIZE = 10_000
MAX_GRID_CELLS = 5_000_000

# Automatic spherical variogram estimation
VARIOGRAM_SAMPLE_POINTS = 5_000
VARIOGRAM_PAIR_COUNT = 80_000
VARIOGRAM_RANDOM_SEED = 42
VARIOGRAM_NUGGET_FRACTION = 0.05

# Numerical stabilizer for local kriging systems
KRIGING_MATRIX_JITTER = 1e-10

# Output settings
NODATA_VALUE = -9999.0
COMPRESS = "lzw"
# ---------------------------------------------------------------------------


def choose_target_crs(gdf: gpd.GeoDataFrame):
    if TARGET_CRS:
        return TARGET_CRS
    if gdf.crs is None:
        raise ValueError("Input shapefile has no CRS. Set TARGET_CRS explicitly.")
    if getattr(gdf.crs, "is_geographic", False):
        utm = gdf.estimate_utm_crs()
        if utm is None:
            raise ValueError(
                "Could not infer projected CRS from geographic input. Set TARGET_CRS."
            )
        return utm
    return gdf.crs


def filter_phase2_excluded_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if not FILTER_EXCLUDED_PHASE2_POINTS:
        return gdf

    if PHASE2_STATUS_COLUMN not in gdf.columns:
        print(
            f"{PHASE2_STATUS_COLUMN} not found; skipping exclusion filter in phase 3."
        )
        return gdf

    status = gdf[PHASE2_STATUS_COLUMN].fillna("").astype(str)
    excluded = np.zeros(len(gdf), dtype=bool)
    for prefix in PHASE2_EXCLUDED_STATUS_PREFIXES:
        excluded |= status.str.startswith(prefix)

    n_excluded = int(excluded.sum())
    if n_excluded > 0:
        print(
            f"Phase-2 exclusion filter: removed {n_excluded} points "
            f"(prefixes={PHASE2_EXCLUDED_STATUS_PREFIXES})"
        )

    kept = gdf.iloc[np.flatnonzero(~excluded)].copy()
    if kept.empty:
        raise ValueError(
            "No points left after phase-2 exclusion filtering; cannot interpolate."
        )
    return kept


def prepare_points(
    gdf: gpd.GeoDataFrame,
) -> tuple[np.ndarray, np.ndarray, gpd.GeoDataFrame]:
    if VALUE_COLUMN not in gdf.columns:
        raise KeyError(f"Missing required column: {VALUE_COLUMN}")

    vals = pd.to_numeric(gdf[VALUE_COLUMN], errors="coerce").to_numpy(dtype=float)
    geom = gdf.geometry

    # Expected input is point samples. For non-point geometries, use representative points.
    if (geom.geom_type == "Point").all():
        px = geom.x.to_numpy(dtype=float)
        py = geom.y.to_numpy(dtype=float)
    else:
        reps = geom.representative_point()
        px = reps.x.to_numpy(dtype=float)
        py = reps.y.to_numpy(dtype=float)

    mask = np.isfinite(vals) & np.isfinite(px) & np.isfinite(py)
    if LOG_TRANSFORM_VALUES:
        mask &= vals > 0
    if not np.any(mask):
        raise ValueError("No finite interpolation points found after filtering.")

    clean = gdf.iloc[np.flatnonzero(mask)].copy()
    points = np.column_stack([px[mask], py[mask]])
    values = vals[mask]

    if LOG_TRANSFORM_VALUES:
        values = np.log(values)
        print("Using natural-log transformed values for kriging.")

    points, values = aggregate_duplicate_points(points, values)
    return points, values, clean


def aggregate_duplicate_points(
    points_xy: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame({"x": points_xy[:, 0], "y": points_xy[:, 1], "z": values})
    grouped = df.groupby(["x", "y"], sort=False, as_index=False)["z"].mean()
    removed = len(df) - len(grouped)
    if removed:
        print(f"Aggregated {removed} duplicate-coordinate samples by mean value.")
    return grouped[["x", "y"]].to_numpy(dtype=float), grouped["z"].to_numpy(dtype=float)


def compute_grid(bounds: tuple[float, float, float, float]) -> tuple[int, int, object]:
    minx, miny, maxx, maxy = bounds
    if not np.isfinite([minx, miny, maxx, maxy]).all():
        raise ValueError("Invalid bounds for interpolation grid.")
    if maxx <= minx or maxy <= miny:
        raise ValueError("Degenerate bounds; cannot build raster grid.")

    ncols = int(math.ceil((maxx - minx) / PIXEL_SIZE))
    nrows = int(math.ceil((maxy - miny) / PIXEL_SIZE))
    total_cells = nrows * ncols
    if total_cells > MAX_GRID_CELLS:
        raise ValueError(
            f"Grid would contain {total_cells:,} cells at PIXEL_SIZE={PIXEL_SIZE:g}. "
            f"Increase PIXEL_SIZE or MAX_GRID_CELLS before running kriging."
        )
    transform = from_origin(minx, maxy, PIXEL_SIZE, PIXEL_SIZE)
    return nrows, ncols, transform


def spherical_semivariogram(
    distance: np.ndarray,
    *,
    nugget: float,
    sill: float,
    range_: float,
) -> np.ndarray:
    h = np.asarray(distance, dtype=float)
    hr = np.clip(h / max(range_, 1e-12), 0.0, None)
    partial_sill = max(sill - nugget, 0.0)
    gamma = np.where(
        h <= 0,
        0.0,
        np.where(
            hr < 1.0,
            nugget + partial_sill * (1.5 * hr - 0.5 * hr**3),
            sill,
        ),
    )
    return gamma


def estimate_spherical_variogram(
    points_xy: np.ndarray,
    values: np.ndarray,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(VARIOGRAM_RANDOM_SEED)
    n = len(values)
    if n < 3:
        raise ValueError("At least 3 points are required for ordinary kriging.")

    sample_n = min(n, int(VARIOGRAM_SAMPLE_POINTS))
    sample_idx = rng.choice(n, size=sample_n, replace=False)
    pts = points_xy[sample_idx]
    vals = values[sample_idx]

    pair_count = min(int(VARIOGRAM_PAIR_COUNT), sample_n * max(sample_n - 1, 1) // 2)
    i = rng.integers(0, sample_n, size=pair_count)
    j = rng.integers(0, sample_n, size=pair_count)
    keep = i != j
    i = i[keep]
    j = j[keep]

    distances = np.linalg.norm(pts[i] - pts[j], axis=1)
    semivariance = 0.5 * (vals[i] - vals[j]) ** 2
    keep = np.isfinite(distances) & np.isfinite(semivariance) & (distances > 0)
    distances = distances[keep]
    semivariance = semivariance[keep]
    if distances.size == 0:
        raise ValueError("Could not estimate a variogram from the input points.")

    sill = float(np.nanvar(values))
    if not np.isfinite(sill) or sill <= 0:
        raise ValueError("Input values have no variance; kriging cannot be fit.")

    nugget = float(np.nanquantile(semivariance, 0.05))
    nugget = float(np.clip(nugget, 0.0, VARIOGRAM_NUGGET_FRACTION * sill))

    target_semivar = nugget + 0.95 * max(sill - nugget, 0.0)
    order = np.argsort(distances)
    sorted_dist = distances[order]
    sorted_semivar = semivariance[order]
    reached = sorted_dist[sorted_semivar >= target_semivar]
    if reached.size:
        range_ = float(np.nanquantile(reached, 0.10))
    else:
        range_ = float(np.nanquantile(distances, 0.80))
    if not np.isfinite(range_) or range_ <= 0:
        range_ = float(np.nanmedian(distances))

    print(
        "Estimated spherical variogram: "
        f"sample_points={sample_n}, pairs={distances.size}, "
        f"nugget={nugget:.6g}, sill={sill:.6g}, range={range_:.6g}"
    )
    return nugget, sill, range_


def ordinary_kriging_predict_one(
    query_xy: np.ndarray,
    neighbor_xy: np.ndarray,
    neighbor_values: np.ndarray,
    *,
    nugget: float,
    sill: float,
    range_: float,
) -> float:
    n = len(neighbor_values)
    if n == 0:
        return np.nan
    if n == 1:
        return float(neighbor_values[0])

    neighbor_dist = np.linalg.norm(
        neighbor_xy[:, None, :] - neighbor_xy[None, :, :],
        axis=2,
    )
    query_dist = np.linalg.norm(neighbor_xy - query_xy[None, :], axis=1)

    matrix = np.empty((n + 1, n + 1), dtype=float)
    matrix[:n, :n] = spherical_semivariogram(
        neighbor_dist,
        nugget=nugget,
        sill=sill,
        range_=range_,
    )
    matrix[:n, n] = 1.0
    matrix[n, :n] = 1.0
    matrix[n, n] = 0.0
    matrix[:n, :n] += np.eye(n) * KRIGING_MATRIX_JITTER

    rhs = np.empty(n + 1, dtype=float)
    rhs[:n] = spherical_semivariogram(
        query_dist,
        nugget=nugget,
        sill=sill,
        range_=range_,
    )
    rhs[n] = 1.0

    try:
        weights = np.linalg.solve(matrix, rhs)[:n]
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(matrix, rhs, rcond=None)[0][:n]

    return float(np.dot(weights, neighbor_values))


def ordinary_kriging_predict_chunk(
    tree: cKDTree,
    points_xy: np.ndarray,
    values: np.ndarray,
    query_points: np.ndarray,
    *,
    nugget: float,
    sill: float,
    range_: float,
) -> np.ndarray:
    k = max(1, int(KRIGING_K_NEIGHBORS))
    k = min(k, len(values))
    dist_upper = float(KRIGING_MAX_DISTANCE) if KRIGING_MAX_DISTANCE > 0 else np.inf

    distances, indices = tree.query(query_points, k=k, distance_upper_bound=dist_upper)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    out = np.full(query_points.shape[0], np.nan, dtype=float)
    for row_idx in range(query_points.shape[0]):
        valid = np.isfinite(distances[row_idx]) & (indices[row_idx] < len(values))
        if int(np.sum(valid)) < KRIGING_MIN_NEIGHBORS:
            continue

        # Exact hits take the exact sampled value.
        exact = valid & (distances[row_idx] == 0.0)
        if np.any(exact):
            out[row_idx] = values[indices[row_idx][np.flatnonzero(exact)[0]]]
            continue

        ii = indices[row_idx][valid]
        out[row_idx] = ordinary_kriging_predict_one(
            query_points[row_idx],
            points_xy[ii],
            values[ii],
            nugget=nugget,
            sill=sill,
            range_=range_,
        )
    return out


def back_transform(values: np.ndarray) -> np.ndarray:
    if not LOG_TRANSFORM_VALUES:
        return values
    return np.exp(values)


def run_interpolation() -> None:
    if cKDTree is None:
        raise ImportError(
            "scipy is required for phase 3 ordinary kriging (cKDTree). "
            "Install scipy and rerun."
        )

    print(f"Reading leveled shapefile: {INPUT_SHP}")
    gdf = gpd.read_file(INPUT_SHP)
    print(f"Rows: {len(gdf)}")
    gdf = filter_phase2_excluded_points(gdf)
    print(f"Rows after phase-2 exclusion filter: {len(gdf)}")

    target_crs = choose_target_crs(gdf)
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
        print(f"Reprojected to {target_crs}")
    else:
        print(f"Using CRS: {target_crs}")

    pts_xy, vals, clean = prepare_points(gdf)
    print(f"Kriging points: {len(vals)}")

    nrows, ncols, transform = compute_grid(tuple(clean.total_bounds))
    total_cells = nrows * ncols
    print(f"Grid: {nrows} x {ncols} ({total_cells} cells), pixel={PIXEL_SIZE}")

    nugget, sill, range_ = estimate_spherical_variogram(pts_xy, vals)
    tree = cKDTree(pts_xy)
    raster_flat = np.full(total_cells, NODATA_VALUE, dtype=np.float32)

    progress = make_progress(total_cells, "Phase 3 ordinary kriging", unit="cell")

    minx, miny, maxx, maxy = clean.total_bounds
    for start in range(0, total_cells, QUERY_CHUNK_SIZE):
        end = min(total_cells, start + QUERY_CHUNK_SIZE)
        idx = np.arange(start, end, dtype=np.int64)

        row = idx // ncols
        col = idx % ncols
        xq = minx + (col + 0.5) * PIXEL_SIZE
        yq = maxy - (row + 0.5) * PIXEL_SIZE
        qpts = np.column_stack([xq, yq])

        pred = ordinary_kriging_predict_chunk(
            tree,
            pts_xy,
            vals,
            qpts,
            nugget=nugget,
            sill=sill,
            range_=range_,
        )
        pred = back_transform(pred)
        chunk = np.where(np.isfinite(pred), pred, NODATA_VALUE).astype(np.float32)
        raster_flat[start:end] = chunk

        progress.update(end - start)

    progress.close()

    raster = raster_flat.reshape((nrows, ncols))

    OUTPUT_TIF.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": nrows,
        "width": ncols,
        "count": 1,
        "dtype": "float32",
        "crs": target_crs,
        "transform": transform,
        "nodata": float(NODATA_VALUE),
        "compress": COMPRESS,
    }

    with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:
        dst.write(raster, 1)

    valid = raster[raster != NODATA_VALUE]
    if valid.size:
        print(
            f"Saved {OUTPUT_TIF} | valid cells={valid.size}, "
            f"min={float(valid.min()):.6g}, max={float(valid.max()):.6g}"
        )
    else:
        print(f"Saved {OUTPUT_TIF} | no valid cells produced")


def main() -> None:
    run_interpolation()


if __name__ == "__main__":
    main()
