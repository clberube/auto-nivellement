from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


class _SimpleProgress:
    def __init__(self, total: int, desc: str, unit: str = "item") -> None:
        self.total = max(int(total), 0)
        self.count = 0
        self.desc = desc
        self.unit = unit
        print(f"{self.desc}: 0/{self.total} {self.unit}")

    def update(self, n: int = 1) -> None:
        self.count = min(self.total, self.count + n)
        pct = (100.0 * self.count / self.total) if self.total else 100.0
        print(
            f"\r{self.desc}: {self.count}/{self.total} {self.unit} ({pct:5.1f}%)",
            end="",
            flush=True,
        )

    def close(self) -> None:
        print()


def make_progress(total: int, desc: str, unit: str = "item"):
    if tqdm is not None:
        return tqdm(total=total, desc=desc, unit=unit)
    return _SimpleProgress(total=total, desc=desc, unit=unit)


def safe_name(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "-_ ." else "_" for ch in text).replace(
        " ", "_"
    )


def validate_required_columns(gdf: gpd.GeoDataFrame, required_columns: set[str]) -> None:
    required = set(required_columns)
    required.add("geometry")
    missing = sorted(required.difference(gdf.columns))
    if missing:
        available = ", ".join(map(str, gdf.columns))
        raise KeyError(
            f"Missing required columns: {missing}. Available columns: {available}"
        )


def normalize_project_ids(
    gdf: gpd.GeoDataFrame,
    project_column: str,
    unknown_label: str = "Unknown",
) -> int:
    project_ids = gdf[project_column].astype("object")
    project_ids = project_ids.where(project_ids.notna(), unknown_label)
    project_ids = project_ids.astype(str).str.strip()
    unknown_tokens = {"", "None", "none", "nan", "NaN", "<NA>"}
    project_ids = project_ids.where(~project_ids.isin(unknown_tokens), unknown_label)
    gdf[project_column] = project_ids
    return int((project_ids == unknown_label).sum())


def coerce_numeric_columns(
    gdf: gpd.GeoDataFrame,
    raw_value_column: str,
    imputed_value_column: str,
    censored_column: str,
) -> tuple[int, int]:
    raw_vals = pd.to_numeric(gdf[raw_value_column], errors="coerce")
    imp_vals = pd.to_numeric(gdf[imputed_value_column], errors="coerce")
    cens_vals = pd.to_numeric(gdf[censored_column], errors="coerce")

    gdf[raw_value_column] = raw_vals
    gdf[imputed_value_column] = imp_vals
    gdf[censored_column] = cens_vals

    return int(raw_vals.isna().sum()), int(imp_vals.isna().sum())


def build_survey_levelability(
    gdf: gpd.GeoDataFrame,
    *,
    project_column: str,
    raw_value_column: str,
    imputed_value_column: str,
    censored_column: str,
    min_pct_above_or_equal_lod: float,
    max_pct_equal_lod: float,
    lod_equality_atol: float,
) -> pd.DataFrame:
    project = gdf[project_column].to_numpy()
    raw = pd.to_numeric(gdf[raw_value_column], errors="coerce").to_numpy(dtype=float)
    imp = pd.to_numeric(gdf[imputed_value_column], errors="coerce").to_numpy(
        dtype=float
    )
    cens = (
        pd.to_numeric(gdf[censored_column], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=float)
        >= 1.0
    )

    lod = np.full(len(gdf), np.nan, dtype=float)
    lod[cens & np.isfinite(raw)] = np.abs(raw[cens & np.isfinite(raw)])

    above_or_equal = np.zeros(len(gdf), dtype=bool)
    non_cens_finite = (~cens) & np.isfinite(imp)
    above_or_equal[non_cens_finite] = True

    cens_valid = cens & np.isfinite(imp) & np.isfinite(lod)
    above_or_equal[cens_valid] = imp[cens_valid] + lod_equality_atol >= lod[cens_valid]

    equal_lod = np.zeros(len(gdf), dtype=bool)
    equal_lod[cens_valid] = np.isclose(
        imp[cens_valid],
        lod[cens_valid],
        atol=lod_equality_atol,
        rtol=0.0,
    )

    row_df = pd.DataFrame(
        {
            project_column: project,
            "imp_finite": np.isfinite(imp),
            "is_censored": cens,
            "has_lod": np.isfinite(lod),
            "above_or_equal_lod": above_or_equal,
            "equal_lod": equal_lod,
        }
    )

    qa = (
        row_df.groupby(project_column)
        .agg(
            sample_count=(project_column, "size"),
            finite_imp_count=("imp_finite", "sum"),
            censored_count=("is_censored", "sum"),
            censored_with_lod_count=("has_lod", "sum"),
            pct_above_or_equal_lod=("above_or_equal_lod", "mean"),
            pct_equal_lod=("equal_lod", "mean"),
        )
        .reset_index()
    )

    qa["levelable"] = (
        qa["pct_above_or_equal_lod"] >= min_pct_above_or_equal_lod
    ) & (qa["pct_equal_lod"] <= max_pct_equal_lod)
    qa["exclude_reason"] = np.where(
        qa["levelable"],
        "",
        "fails_lod_rules",
    )
    return qa


def build_project_counts(
    gdf: gpd.GeoDataFrame,
    project_column: str,
    projects: set[str],
) -> pd.Series:
    counts = gdf[project_column].value_counts()
    counts = counts[counts.index.isin(projects)]
    return counts.sort_values(ascending=False)


def build_project_footprints(
    gdf: gpd.GeoDataFrame,
    project_column: str,
    projects: set[str],
) -> dict[str, object]:
    footprints: dict[str, object] = {}
    for project_id, part in gdf.groupby(project_column, sort=False):
        if project_id not in projects or part.empty:
            continue
        geom_series = part.geometry
        if hasattr(geom_series, "union_all"):
            merged = geom_series.union_all()
        else:
            merged = geom_series.unary_union
        footprint = merged.convex_hull
        if not footprint.is_empty:
            footprints[project_id] = footprint
    return footprints


def is_fully_overlapped(candidate_footprint, reference_footprint) -> bool:
    if candidate_footprint is None or reference_footprint is None:
        return False
    if candidate_footprint.is_empty or reference_footprint.is_empty:
        return False
    return bool(reference_footprint.covers(candidate_footprint))


def pair_values_by_rank(
    ref_vals: np.ndarray,
    cand_vals: np.ndarray,
    *,
    min_pair_count: int,
    max_pair_count: int,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    ref_vals = ref_vals[np.isfinite(ref_vals)]
    cand_vals = cand_vals[np.isfinite(cand_vals)]
    if len(ref_vals) == 0 or len(cand_vals) == 0:
        return None, None

    n = int(min(len(ref_vals), len(cand_vals), max_pair_count))
    if n < min_pair_count:
        return None, None

    q = np.linspace(0.0, 1.0, n)
    ref_rank = np.quantile(ref_vals, q)
    cand_rank = np.quantile(cand_vals, q)
    return ref_rank.astype(float), cand_rank.astype(float)


def trim_pairs(
    ref_vals: np.ndarray,
    cand_vals: np.ndarray,
    *,
    trim_lower_quantile: float,
    trim_upper_quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isfinite(ref_vals) & np.isfinite(cand_vals)
    ref_vals = ref_vals[keep]
    cand_vals = cand_vals[keep]
    if len(ref_vals) == 0:
        return ref_vals, cand_vals

    ref_lo, ref_hi = np.quantile(ref_vals, [trim_lower_quantile, trim_upper_quantile])
    cand_lo, cand_hi = np.quantile(cand_vals, [trim_lower_quantile, trim_upper_quantile])

    keep = (
        (ref_vals >= ref_lo)
        & (ref_vals <= ref_hi)
        & (cand_vals >= cand_lo)
        & (cand_vals <= cand_hi)
    )
    return ref_vals[keep], cand_vals[keep]


def stable_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_variance: float = 1e-16,
) -> tuple[float, float]:
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]

    if len(x) < 2:
        return 1.0, float(np.mean(y) - np.mean(x))

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_centered = x - x_mean
    y_centered = y - y_mean
    var_x = float(np.mean(x_centered * x_centered))
    var_y = float(np.mean(y_centered * y_centered))
    cov_xy = float(np.mean(x_centered * y_centered))

    if (
        var_x <= min_variance
        or var_y <= min_variance
        or not np.isfinite(cov_xy)
        or abs(cov_xy) <= min_variance
    ):
        return 1.0, y_mean - x_mean

    cov_mat = np.array([[var_x, cov_xy], [cov_xy, var_y]], dtype=float)
    eigvals, eigvecs = np.linalg.eigh(cov_mat)
    v = eigvecs[:, int(np.argmax(eigvals))]
    vx = float(v[0])
    vy = float(v[1])
    if abs(vx) <= min_variance or not np.isfinite(vx) or not np.isfinite(vy):
        return 1.0, y_mean - x_mean

    slope = vy / vx
    intercept = y_mean - slope * x_mean
    return slope, intercept


def fit_correction(
    ref_vals: np.ndarray,
    cand_vals: np.ndarray,
    *,
    min_variance: float = 1e-16,
) -> tuple[float, float, float, float]:
    slope, intercept = stable_linear_fit(cand_vals, ref_vals, min_variance=min_variance)
    corrected = slope * cand_vals + intercept
    post_slope, post_intercept = stable_linear_fit(
        corrected,
        ref_vals,
        min_variance=min_variance,
    )
    return float(slope), float(intercept), float(post_slope), float(post_intercept)


def save_regression_plot(
    plot_path: Path,
    fit_ref: np.ndarray,
    fit_cand: np.ndarray,
    slope: float,
    intercept: float,
    post_slope: float,
    post_intercept: float,
    reference_project: object,
    candidate_project: object,
    full_slope: float | None = None,
    full_intercept: float | None = None,
) -> None:
    if plt is None:
        return

    corrected = slope * fit_cand + intercept
    all_vals = np.concatenate([fit_cand, corrected, fit_ref])
    x_min = float(np.nanmin(all_vals))
    x_max = float(np.nanmax(all_vals))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(fit_cand, fit_ref, s=12, alpha=0.35, color="#2c7fb8", label="trimmed pre")
    ax.scatter(corrected, fit_ref, s=12, alpha=0.55, color="#1a9850", label="trimmed post")

    if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
        x_line = np.linspace(x_min, x_max, 100)
        if full_slope is not None and full_intercept is not None:
            ax.plot(
                x_line,
                full_slope * x_line + full_intercept,
                color="#d7301f",
                lw=1.8,
                label="full fit",
            )
            ax.plot(
                x_line,
                slope * x_line + intercept,
                color="#fdae61",
                lw=1.8,
                label="applied partial fit",
            )
        else:
            ax.plot(
                x_line,
                slope * x_line + intercept,
                color="#d7301f",
                lw=1.8,
                label="fit",
            )
        ax.plot(x_line, x_line, color="black", lw=1.2, ls="--", label="1:1")

    title_text = f"ref={reference_project} cand={candidate_project}\n"
    if full_slope is not None and full_intercept is not None:
        title_text += (
            f"full=({full_slope:.4f}, {full_intercept:.4f}) "
            f"applied=({slope:.4f}, {intercept:.4f}) "
        )
    else:
        title_text += f"fit=({slope:.4f}, {intercept:.4f}) "
    title_text += f"post=({post_slope:.4f}, {post_intercept:.4f})"
    ax.set_title(title_text)
    ax.set_xlabel("Candidate")
    ax.set_ylabel("Reference")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def apply_correction(
    gdf: gpd.GeoDataFrame,
    *,
    project_column: str,
    value_column: str,
    project_id: object,
    slope: float,
    intercept: float,
    step: int,
    reference_project: object,
) -> None:
    mask = gdf[project_column] == project_id
    gdf.loc[mask, value_column] = gdf.loc[mask, value_column] * slope + intercept
    gdf.loc[mask, "LVL_MUL"] = gdf.loc[mask, "LVL_MUL"] * slope
    gdf.loc[mask, "LVL_ADD"] = gdf.loc[mask, "LVL_ADD"] * slope + intercept
    gdf.loc[mask, "LVL_STP"] = step
    gdf.loc[mask, "LVL_REF"] = str(reference_project)
    gdf.loc[mask, "LVL_STAT"] = "leveled"
