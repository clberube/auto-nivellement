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

# -------------------------- User-editable settings --------------------------
INPUT_SHP = Path("shp/AG_Fusionn_imp.shp")
OUTPUT_DIR = Path("output")

PROJECT_COLUMN = "NUMR_PROJ_"
RAW_VALUE_COLUMN = "Ag"
IMPUTED_VALUE_COLUMN = "Ag_imp"
CENSORED_COLUMN = "is_censor"
LITHOLOGY_COLUMN = "CODE_TYPE_"

# Survey-level levelability rules
MIN_PCT_ABOVE_OR_EQUAL_LOD = 0.70
MAX_PCT_EQUAL_LOD = 0.30
LOD_EQUALITY_ATOL = 1e-12

# Pairing / regression
MIN_PAIR_COUNT = 30
MAX_PAIR_COUNT = 1000
MIN_CANDIDATE_UNIQUE_VALUES = 3
TRIM_LOWER_QUANTILE = 0.1
TRIM_UPPER_QUANTILE = 0.9
LINEAR_FIT_MIN_VARIANCE = 1e-16

# Outputs
SAVE_REGRESSION_PLOTS = True
REGRESSION_PLOTS_DIRNAME = "regression_plots"
FINAL_OUTPUT_NAME = "leveled_full_overlap.shp"
LOG_CSV_NAME = "leveling_log.csv"
SURVEY_QA_CSV_NAME = "survey_levelability.csv"
EXCLUDED_SURVEYS_CSV_NAME = "excluded_surveys.csv"
# ---------------------------------------------------------------------------


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


def validate_required_columns(gdf: gpd.GeoDataFrame) -> None:
    required = {
        PROJECT_COLUMN,
        RAW_VALUE_COLUMN,
        IMPUTED_VALUE_COLUMN,
        CENSORED_COLUMN,
        LITHOLOGY_COLUMN,
        "geometry",
    }
    missing = sorted(required.difference(gdf.columns))
    if missing:
        available = ", ".join(map(str, gdf.columns))
        raise KeyError(
            f"Missing required columns: {missing}. Available columns: {available}"
        )


def normalize_project_ids(gdf: gpd.GeoDataFrame) -> int:
    project_ids = gdf[PROJECT_COLUMN].astype("object")
    project_ids = project_ids.where(project_ids.notna(), "Unknown")
    project_ids = project_ids.astype(str).str.strip()
    unknown_tokens = {"", "None", "none", "nan", "NaN", "<NA>"}
    project_ids = project_ids.where(~project_ids.isin(unknown_tokens), "Unknown")
    gdf[PROJECT_COLUMN] = project_ids
    return int((project_ids == "Unknown").sum())


def coerce_numeric_columns(gdf: gpd.GeoDataFrame) -> tuple[int, int]:
    raw_vals = pd.to_numeric(gdf[RAW_VALUE_COLUMN], errors="coerce")
    imp_vals = pd.to_numeric(gdf[IMPUTED_VALUE_COLUMN], errors="coerce")
    cens_vals = pd.to_numeric(gdf[CENSORED_COLUMN], errors="coerce")

    gdf[RAW_VALUE_COLUMN] = raw_vals
    gdf[IMPUTED_VALUE_COLUMN] = imp_vals
    gdf[CENSORED_COLUMN] = cens_vals

    return int(raw_vals.isna().sum()), int(imp_vals.isna().sum())


def build_survey_levelability(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    project = gdf[PROJECT_COLUMN].to_numpy()
    raw = pd.to_numeric(gdf[RAW_VALUE_COLUMN], errors="coerce").to_numpy(dtype=float)
    imp = pd.to_numeric(gdf[IMPUTED_VALUE_COLUMN], errors="coerce").to_numpy(
        dtype=float
    )
    cens = (
        pd.to_numeric(gdf[CENSORED_COLUMN], errors="coerce")
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
    above_or_equal[cens_valid] = imp[cens_valid] + LOD_EQUALITY_ATOL >= lod[cens_valid]

    equal_lod = np.zeros(len(gdf), dtype=bool)
    equal_lod[cens_valid] = np.isclose(
        imp[cens_valid],
        lod[cens_valid],
        atol=LOD_EQUALITY_ATOL,
        rtol=0.0,
    )

    row_df = pd.DataFrame(
        {
            PROJECT_COLUMN: project,
            "imp_finite": np.isfinite(imp),
            "is_censored": cens,
            "has_lod": np.isfinite(lod),
            "above_or_equal_lod": above_or_equal,
            "equal_lod": equal_lod,
        }
    )

    qa = (
        row_df.groupby(PROJECT_COLUMN)
        .agg(
            sample_count=(PROJECT_COLUMN, "size"),
            finite_imp_count=("imp_finite", "sum"),
            censored_count=("is_censored", "sum"),
            censored_with_lod_count=("has_lod", "sum"),
            pct_above_or_equal_lod=("above_or_equal_lod", "mean"),
            pct_equal_lod=("equal_lod", "mean"),
        )
        .reset_index()
    )

    qa["levelable"] = (qa["pct_above_or_equal_lod"] >= MIN_PCT_ABOVE_OR_EQUAL_LOD) & (
        qa["pct_equal_lod"] <= MAX_PCT_EQUAL_LOD
    )
    qa["exclude_reason"] = np.where(
        qa["levelable"],
        "",
        "fails_lod_rules",
    )
    return qa


def build_project_counts(gdf: gpd.GeoDataFrame, projects: set[str]) -> pd.Series:
    counts = gdf[PROJECT_COLUMN].value_counts()
    counts = counts[counts.index.isin(projects)]
    return counts.sort_values(ascending=False)


def build_project_footprints(
    gdf: gpd.GeoDataFrame, projects: set[str]
) -> dict[str, object]:
    footprints: dict[str, object] = {}
    for project_id, part in gdf.groupby(PROJECT_COLUMN, sort=False):
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
    ref_vals: np.ndarray, cand_vals: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    ref_vals = ref_vals[np.isfinite(ref_vals)]
    cand_vals = cand_vals[np.isfinite(cand_vals)]
    if len(ref_vals) == 0 or len(cand_vals) == 0:
        return None, None

    n = int(min(len(ref_vals), len(cand_vals), MAX_PAIR_COUNT))
    if n < MIN_PAIR_COUNT:
        return None, None

    q = np.linspace(0.0, 1.0, n)
    ref_rank = np.quantile(ref_vals, q)
    cand_rank = np.quantile(cand_vals, q)
    return ref_rank.astype(float), cand_rank.astype(float)


def trim_pairs(
    ref_vals: np.ndarray, cand_vals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isfinite(ref_vals) & np.isfinite(cand_vals)
    ref_vals = ref_vals[keep]
    cand_vals = cand_vals[keep]
    if len(ref_vals) == 0:
        return ref_vals, cand_vals

    ref_lo, ref_hi = np.quantile(ref_vals, [TRIM_LOWER_QUANTILE, TRIM_UPPER_QUANTILE])
    cand_lo, cand_hi = np.quantile(
        cand_vals, [TRIM_LOWER_QUANTILE, TRIM_UPPER_QUANTILE]
    )

    keep = (
        (ref_vals >= ref_lo)
        & (ref_vals <= ref_hi)
        & (cand_vals >= cand_lo)
        & (cand_vals <= cand_hi)
    )
    return ref_vals[keep], cand_vals[keep]


def stable_linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
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

    # Principal-axes regression (major axis): first eigenvector of covariance matrix.
    # This avoids selecting one variable as dependent/independent.
    if (
        var_x <= LINEAR_FIT_MIN_VARIANCE
        or var_y <= LINEAR_FIT_MIN_VARIANCE
        or not np.isfinite(cov_xy)
        or abs(cov_xy) <= LINEAR_FIT_MIN_VARIANCE
    ):
        return 1.0, y_mean - x_mean

    cov_mat = np.array([[var_x, cov_xy], [cov_xy, var_y]], dtype=float)
    eigvals, eigvecs = np.linalg.eigh(cov_mat)
    v = eigvecs[:, int(np.argmax(eigvals))]
    vx = float(v[0])
    vy = float(v[1])
    if abs(vx) <= LINEAR_FIT_MIN_VARIANCE or not np.isfinite(vx) or not np.isfinite(vy):
        return 1.0, y_mean - x_mean

    slope = vy / vx
    intercept = y_mean - slope * x_mean
    return slope, intercept


def fit_correction(
    ref_vals: np.ndarray, cand_vals: np.ndarray
) -> tuple[float, float, float, float]:
    slope, intercept = stable_linear_fit(cand_vals, ref_vals)
    corrected = slope * cand_vals + intercept
    post_slope, post_intercept = stable_linear_fit(corrected, ref_vals)
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
) -> None:
    if plt is None:
        return

    corrected = slope * fit_cand + intercept
    all_vals = np.concatenate([fit_cand, corrected, fit_ref])
    x_min = float(np.nanmin(all_vals))
    x_max = float(np.nanmax(all_vals))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        fit_cand, fit_ref, s=12, alpha=0.35, color="#2c7fb8", label="trimmed pre"
    )
    ax.scatter(
        corrected, fit_ref, s=12, alpha=0.55, color="#1a9850", label="trimmed post"
    )

    if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
        x_line = np.linspace(x_min, x_max, 100)
        ax.plot(
            x_line, slope * x_line + intercept, color="#d7301f", lw=1.8, label="pre fit"
        )
        ax.plot(x_line, x_line, color="black", lw=1.2, ls="--", label="1:1")

    ax.set_title(
        f"ref={reference_project} cand={candidate_project}\n"
        f"pre=({slope:.4f}, {intercept:.4f}) post=({post_slope:.4f}, {post_intercept:.4f})"
    )
    ax.set_xlabel("Candidate")
    ax.set_ylabel("Reference")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def apply_correction(
    gdf: gpd.GeoDataFrame,
    project_id: object,
    slope: float,
    intercept: float,
    step: int,
    reference_project: object,
) -> None:
    mask = gdf[PROJECT_COLUMN] == project_id
    gdf.loc[mask, IMPUTED_VALUE_COLUMN] = (
        gdf.loc[mask, IMPUTED_VALUE_COLUMN] * slope + intercept
    )
    gdf.loc[mask, "LVL_MUL"] = gdf.loc[mask, "LVL_MUL"] * slope
    gdf.loc[mask, "LVL_ADD"] = gdf.loc[mask, "LVL_ADD"] * slope + intercept
    gdf.loc[mask, "LVL_STP"] = step
    gdf.loc[mask, "LVL_REF"] = str(reference_project)
    gdf.loc[mask, "LVL_STAT"] = "leveled"


def run_leveling(
    gdf: gpd.GeoDataFrame,
    survey_qa: pd.DataFrame,
    final_output_name: str,
    log_csv_name: str,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    levelable_projects = set(survey_qa.loc[survey_qa["levelable"], PROJECT_COLUMN])
    excluded_quality_projects = set(
        survey_qa.loc[~survey_qa["levelable"], PROJECT_COLUMN]
    )

    counts = build_project_counts(gdf, levelable_projects)
    if counts.empty:
        raise ValueError("No surveys passed the levelability rules.")

    ordered_projects = list(counts.index)
    footprints = build_project_footprints(gdf, levelable_projects)
    # GroupBy.indices are positional indices, so we must use iloc.
    project_indices = gdf.groupby(PROJECT_COLUMN, sort=False).indices

    gdf["LVL_MUL"] = 1.0
    gdf["LVL_ADD"] = 0.0
    gdf["LVL_STP"] = 0
    gdf["LVL_REF"] = ""
    gdf["LVL_STAT"] = "eligible_unleveled"
    if excluded_quality_projects:
        gdf.loc[gdf[PROJECT_COLUMN].isin(excluded_quality_projects), "LVL_STAT"] = (
            "excluded_quality"
        )

    plot_dir = OUTPUT_DIR / REGRESSION_PLOTS_DIRNAME
    if SAVE_REGRESSION_PLOTS and plt is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_REGRESSION_PLOTS and plt is None:
        print("Regression plots requested, but matplotlib is not available.")

    corrected_projects: set[str] = set()
    excluded_low_unique_values: set[str] = set()
    excluded_insufficient_pairs: set[str] = set()
    log_rows: list[dict] = []
    step = 0

    progress = make_progress(
        len(ordered_projects), "Containment leveling", unit="survey"
    )

    for reference_project in ordered_projects:
        if reference_project in excluded_insufficient_pairs:
            progress.update(1)
            continue
        if reference_project in excluded_low_unique_values:
            progress.update(1)
            continue
        if (
            reference_project not in footprints
            or reference_project not in project_indices
        ):
            progress.update(1)
            continue

        ref_fp = footprints[reference_project]
        ref_points = gdf.iloc[project_indices[reference_project]]
        ref_vals = pd.to_numeric(
            ref_points[IMPUTED_VALUE_COLUMN], errors="coerce"
        ).to_numpy(dtype=float)
        ref_unique_count = int(np.unique(ref_vals[np.isfinite(ref_vals)]).size)
        if ref_unique_count < MIN_CANDIDATE_UNIQUE_VALUES:
            excluded_low_unique_values.add(reference_project)
            gdf.loc[gdf[PROJECT_COLUMN] == reference_project, "LVL_STAT"] = (
                "excluded_low_unique_values"
            )
            log_rows.append(
                {
                    "step": np.nan,
                    "status": "excluded_low_unique_values",
                    "reference_project": reference_project,
                    "candidate_project": np.nan,
                    "reference_unique_count": ref_unique_count,
                    "candidate_unique_count": np.nan,
                    "pair_count_raw": 0,
                    "pair_count_trimmed": 0,
                    "slope_applied": np.nan,
                    "intercept_applied": np.nan,
                    "post_slope": np.nan,
                    "post_intercept": np.nan,
                    "regression_plot": "",
                }
            )
            progress.update(1)
            continue

        for candidate in ordered_projects:
            if candidate == reference_project:
                continue
            if candidate in corrected_projects:
                continue
            if candidate in excluded_insufficient_pairs:
                continue
            if candidate in excluded_low_unique_values:
                continue
            if counts.get(candidate, 0) >= counts.get(reference_project, 0):
                continue
            if candidate not in footprints or candidate not in project_indices:
                continue
            if not is_fully_overlapped(footprints[candidate], ref_fp):
                continue

            cand_points = gdf.iloc[project_indices[candidate]]
            cand_vals = pd.to_numeric(
                cand_points[IMPUTED_VALUE_COLUMN], errors="coerce"
            ).to_numpy(dtype=float)

            ref_pairs, cand_pairs = pair_values_by_rank(ref_vals, cand_vals)
            if ref_pairs is None:
                excluded_insufficient_pairs.add(candidate)
                gdf.loc[gdf[PROJECT_COLUMN] == candidate, "LVL_STAT"] = (
                    "excluded_insufficient_pairs"
                )
                log_rows.append(
                    {
                        "step": np.nan,
                        "status": "excluded_insufficient_pairs",
                        "reference_project": reference_project,
                        "candidate_project": candidate,
                        "candidate_unique_count": np.nan,
                        "pair_count_raw": 0,
                        "pair_count_trimmed": 0,
                        "slope_applied": np.nan,
                        "intercept_applied": np.nan,
                        "post_slope": np.nan,
                        "post_intercept": np.nan,
                        "regression_plot": "",
                    }
                )
                continue

            raw_count = int(len(ref_pairs))
            fit_ref, fit_cand = trim_pairs(ref_pairs, cand_pairs)
            fit_count = int(len(fit_ref))
            ref_trimmed_unique_count = int(
                np.unique(fit_ref[np.isfinite(fit_ref)]).size
            )
            cand_trimmed_unique_count = int(
                np.unique(fit_cand[np.isfinite(fit_cand)]).size
            )
            if fit_count < MIN_PAIR_COUNT:
                excluded_insufficient_pairs.add(candidate)
                gdf.loc[gdf[PROJECT_COLUMN] == candidate, "LVL_STAT"] = (
                    "excluded_insufficient_pairs"
                )
                log_rows.append(
                    {
                        "step": np.nan,
                        "status": "excluded_insufficient_pairs",
                        "reference_project": reference_project,
                        "candidate_project": candidate,
                        "candidate_unique_count": cand_trimmed_unique_count,
                        "pair_count_raw": raw_count,
                        "pair_count_trimmed": fit_count,
                        "slope_applied": np.nan,
                        "intercept_applied": np.nan,
                        "post_slope": np.nan,
                        "post_intercept": np.nan,
                        "regression_plot": "",
                    }
                )
                continue

            if ref_trimmed_unique_count < MIN_CANDIDATE_UNIQUE_VALUES:
                excluded_insufficient_pairs.add(candidate)
                gdf.loc[gdf[PROJECT_COLUMN] == candidate, "LVL_STAT"] = (
                    "excluded_insufficient_pairs"
                )
                log_rows.append(
                    {
                        "step": np.nan,
                        "status": "excluded_insufficient_pairs",
                        "reference_project": reference_project,
                        "candidate_project": candidate,
                        "candidate_unique_count": cand_trimmed_unique_count,
                        "pair_count_raw": raw_count,
                        "pair_count_trimmed": fit_count,
                        "slope_applied": np.nan,
                        "intercept_applied": np.nan,
                        "post_slope": np.nan,
                        "post_intercept": np.nan,
                        "regression_plot": "",
                    }
                )
                continue

            if cand_trimmed_unique_count < MIN_CANDIDATE_UNIQUE_VALUES:
                excluded_low_unique_values.add(candidate)
                gdf.loc[gdf[PROJECT_COLUMN] == candidate, "LVL_STAT"] = (
                    "excluded_low_unique_values"
                )
                log_rows.append(
                    {
                        "step": np.nan,
                        "status": "excluded_low_unique_values",
                        "reference_project": reference_project,
                        "candidate_project": candidate,
                        "candidate_unique_count": cand_trimmed_unique_count,
                        "pair_count_raw": raw_count,
                        "pair_count_trimmed": fit_count,
                        "slope_applied": np.nan,
                        "intercept_applied": np.nan,
                        "post_slope": np.nan,
                        "post_intercept": np.nan,
                        "regression_plot": "",
                    }
                )
                continue

            slope, intercept, post_slope, post_intercept = fit_correction(
                fit_ref, fit_cand
            )

            step += 1
            apply_correction(gdf, candidate, slope, intercept, step, reference_project)
            corrected_projects.add(candidate)

            plot_rel = ""
            if SAVE_REGRESSION_PLOTS and plt is not None:
                plot_name = (
                    f"step_{step:04d}__ref_{safe_name(reference_project)}"
                    f"__cand_{safe_name(candidate)}.png"
                )
                plot_path = plot_dir / plot_name
                save_regression_plot(
                    plot_path=plot_path,
                    fit_ref=fit_ref,
                    fit_cand=fit_cand,
                    slope=slope,
                    intercept=intercept,
                    post_slope=post_slope,
                    post_intercept=post_intercept,
                    reference_project=reference_project,
                    candidate_project=candidate,
                )
                plot_rel = str(plot_path.relative_to(OUTPUT_DIR))

            log_rows.append(
                {
                    "step": step,
                    "status": "leveled",
                    "reference_project": reference_project,
                    "candidate_project": candidate,
                    "candidate_unique_count": cand_trimmed_unique_count,
                    "pair_count_raw": raw_count,
                    "pair_count_trimmed": fit_count,
                    "slope_applied": slope,
                    "intercept_applied": intercept,
                    "post_slope": post_slope,
                    "post_intercept": post_intercept,
                    "regression_plot": plot_rel,
                }
            )

            print(
                f"Step {step}: leveled {candidate} using {reference_project} "
                f"(pairs={fit_count}/{raw_count}, slope={slope:.4f}, intercept={intercept:.4f})"
            )

        progress.update(1)

    progress.close()

    output_projects = (
        set(levelable_projects)
        .difference(excluded_insufficient_pairs)
        .difference(excluded_low_unique_values)
    )
    final_out = gdf[gdf[PROJECT_COLUMN].isin(output_projects)].copy()

    excluded_df = survey_qa[[PROJECT_COLUMN, "exclude_reason"]].copy()
    if excluded_low_unique_values:
        extra = pd.DataFrame(
            {
                PROJECT_COLUMN: sorted(excluded_low_unique_values),
                "exclude_reason": "low_unique_values",
            }
        )
        excluded_df = pd.concat([excluded_df, extra], ignore_index=True)
    if excluded_insufficient_pairs:
        extra = pd.DataFrame(
            {
                PROJECT_COLUMN: sorted(excluded_insufficient_pairs),
                "exclude_reason": "insufficient_pairs",
            }
        )
        excluded_df = pd.concat([excluded_df, extra], ignore_index=True)
    excluded_df = excluded_df[excluded_df["exclude_reason"] != ""].drop_duplicates(
        subset=[PROJECT_COLUMN], keep="last"
    )

    final_out_path = OUTPUT_DIR / final_output_name
    final_out.to_file(final_out_path)

    log_df = pd.DataFrame(log_rows)
    log_path = OUTPUT_DIR / log_csv_name
    log_df.to_csv(log_path, index=False)

    excluded_path = OUTPUT_DIR / EXCLUDED_SURVEYS_CSV_NAME
    excluded_df.to_csv(excluded_path, index=False)

    print(f"Saved final leveled shapefile: {final_out_path}")
    print(f"Saved leveling log: {log_path}")
    print(f"Saved excluded surveys: {excluded_path}")
    print(
        f"Summary: total_surveys={len(survey_qa)}, "
        f"levelable={len(levelable_projects)}, "
        f"excluded_quality={len(excluded_quality_projects)}, "
        f"excluded_low_unique_values={len(excluded_low_unique_values)}, "
        f"excluded_insufficient_pairs={len(excluded_insufficient_pairs)}, "
        f"leveled={int((log_df['status'] == 'leveled').sum()) if not log_df.empty else 0}"
    )


def main() -> None:
    print(f"Reading shapefile: {INPUT_SHP}")
    gdf = gpd.read_file(INPUT_SHP)
    print(f"Rows: {len(gdf)}")
    print("Columns:", ", ".join(map(str, gdf.columns)))

    validate_required_columns(gdf)
    unknown_count = normalize_project_ids(gdf)
    if unknown_count:
        print(f"Normalized missing/None project IDs to 'Unknown': {unknown_count} rows")

    raw_nan_count, imp_nan_count = coerce_numeric_columns(gdf)
    print(
        f"Coerced numeric columns. NaN counts -> {RAW_VALUE_COLUMN}: {raw_nan_count}, "
        f"{IMPUTED_VALUE_COLUMN}: {imp_nan_count}"
    )

    survey_qa = build_survey_levelability(gdf)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    qa_path = OUTPUT_DIR / SURVEY_QA_CSV_NAME
    survey_qa.to_csv(qa_path, index=False)
    print(f"Saved survey levelability QA: {qa_path}")

    run_leveling(
        gdf=gdf,
        survey_qa=survey_qa,
        final_output_name=FINAL_OUTPUT_NAME,
        log_csv_name=LOG_CSV_NAME,
    )


if __name__ == "__main__":
    main()
