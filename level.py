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
# Columns currently present in shp/AG_Fusionn.shp:
# OBJECTID, NUMR_ECHN_, NUMR_FEUIL, CODE_TYPE_, DATE_ECHN, FUS, ESTN, NORD,
# CODE_PREC_, DATE_DERN_, CODE_INDC_, NUMR_INTER, CODE_SYMBL, Ag, NUMR_PROJ_,
# geometry
INPUT_SHP = Path("shp/AG_Fusionn.shp")
OUTPUT_DIR = Path("output")

PROJECT_COLUMN = "NUMR_PROJ_"
LITHOLOGY_COLUMN = "CODE_TYPE_"
VALUE_COLUMN = "Ag"
ISLAND_COLUMN = "ISLAND_ID"
LOW_CONF_COLUMN = "LOW_CONF"

MIN_TOTAL_PAIR_COUNT = 20
MAX_PAIR_COUNT = 1000
RANDOM_SEED = 42
LINEAR_FIT_MIN_VARIANCE = 1e-16
REGRESSION_LOWER_QUANTILE = 0.01
REGRESSION_UPPER_QUANTILE = 0.99
MIN_TRIMMED_PAIR_COUNT = 16
LOW_CONF_SLOPE_MIN = 0.2
LOW_CONF_SLOPE_MAX = 5.0
SPATIAL_PAIRING_MODE = "rank_based"
MIN_UNIQUE_VALUES_PER_SURVEY = 10  # Enforced on trimmed regression pairs.
# Keep False for production runs (single mode), set True only for alternate setting tests.
RUN_DISTANCE_COMPARISON = False
DISTANCE_COMPARISON_MODES = ("rank_based",)
COMPARISON_METRICS_CSV_NAME = "leveling_comparison_metrics.csv"
RUN_PHASE_2 = False  # Phase 2 is intentionally disabled in this simplified workflow.
OUTPUT_UPPER_PERCENTILE = 0.99
IMPUTE_NEGATIVE_AS_HALF_ABS = True  # Example: -1 -> 0.5, -2 -> 1.0
SAVE_REGRESSION_PLOTS = True
REGRESSION_PLOTS_DIRNAME = "regression_plots"
SAVE_SURVEY_HISTOGRAMS = True
SURVEY_HISTOGRAMS_DIRNAME = "survey_histograms"
SURVEY_HISTOGRAM_BINS = 40
SURVEY_HIST_SUMMARY_CSV_NAME = "survey_histogram_summary.csv"
LOW_UNIQUE_SURVEYS_CSV_NAME = "excluded_low_unique_surveys.csv"

FINAL_OUTPUT_NAME = "leveled_all.shp"
LOG_CSV_NAME = "leveling_log.csv"
# ---------------------------------------------------------------------------


class _SimpleProgress:
    def __init__(self, total: int, desc: str, unit: str = "item") -> None:
        self.total = max(int(total), 0)
        self.count = 0
        self.desc = desc
        self.unit = unit
        self.postfix = ""
        print(f"{self.desc}: 0/{self.total} {self.unit}")

    def set_postfix_str(self, text: str) -> None:
        self.postfix = text

    def update(self, n: int = 1) -> None:
        self.count = min(self.total, self.count + n)
        pct = (100.0 * self.count / self.total) if self.total else 100.0
        suffix = f" | {self.postfix}" if self.postfix else ""
        print(
            f"\r{self.desc}: {self.count}/{self.total} {self.unit} ({pct:5.1f}%){suffix}",
            end="",
            flush=True,
        )

    def close(self) -> None:
        print()


def make_progress(total: int, desc: str, unit: str = "item"):
    if tqdm is not None:
        return tqdm(total=total, desc=desc, unit=unit)
    return _SimpleProgress(total=total, desc=desc, unit=unit)


def _safe_name(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)


def validate_required_columns(gdf: gpd.GeoDataFrame) -> None:
    required = {PROJECT_COLUMN, LITHOLOGY_COLUMN, VALUE_COLUMN, "geometry"}
    missing = sorted(required.difference(gdf.columns))
    if missing:
        available = ", ".join(map(str, gdf.columns))
        raise KeyError(
            f"Missing required columns: {missing}. Available columns: {available}"
        )


def impute_bdl_values(gdf: gpd.GeoDataFrame) -> tuple[int, int]:
    numeric_values = pd.to_numeric(gdf[VALUE_COLUMN], errors="coerce")
    neg_mask = numeric_values < 0

    if IMPUTE_NEGATIVE_AS_HALF_ABS:
        numeric_values.loc[neg_mask] = (-numeric_values.loc[neg_mask]) / 2.0

    gdf[VALUE_COLUMN] = numeric_values
    return int(neg_mask.sum()), int(numeric_values.isna().sum())


def normalize_project_ids(gdf: gpd.GeoDataFrame) -> int:
    project_ids = gdf[PROJECT_COLUMN].astype("object")
    project_ids = project_ids.where(project_ids.notna(), "Unknown")
    project_ids = project_ids.astype(str).str.strip()
    unknown_tokens = {"", "None", "none", "nan", "NaN", "<NA>"}
    project_ids = project_ids.where(~project_ids.isin(unknown_tokens), "Unknown")
    gdf[PROJECT_COLUMN] = project_ids
    return int((project_ids == "Unknown").sum())


def project_counts(gdf: gpd.GeoDataFrame) -> pd.Series:
    return gdf[PROJECT_COLUMN].dropna().value_counts()


def survey_uniqueness_summary(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    values = pd.to_numeric(gdf[VALUE_COLUMN], errors="coerce")
    tmp = pd.DataFrame(
        {
            PROJECT_COLUMN: gdf[PROJECT_COLUMN].to_numpy(),
            VALUE_COLUMN: values.to_numpy(),
        }
    )
    out = (
        tmp.groupby(PROJECT_COLUMN)[VALUE_COLUMN]
        .agg(
            sample_count="size",
            finite_count=lambda s: int(s.notna().sum()),
            unique_count=lambda s: int(s.dropna().nunique()),
        )
        .reset_index()
    )
    out["is_low_unique"] = out["unique_count"] < int(MIN_UNIQUE_VALUES_PER_SURVEY)
    return out


def export_survey_histograms(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    if plt is None:
        print("Survey histogram export requested, but matplotlib is not available.")
        return pd.DataFrame()

    hist_dir = OUTPUT_DIR / SURVEY_HISTOGRAMS_DIRNAME
    hist_dir.mkdir(parents=True, exist_ok=True)

    grouped = gdf.groupby(PROJECT_COLUMN, sort=False)
    progress = make_progress(len(grouped), "Survey histograms", unit="survey")
    summary_rows = []

    for project_id, part in grouped:
        values = pd.to_numeric(part[VALUE_COLUMN], errors="coerce").to_numpy(
            dtype=float
        )
        values = values[np.isfinite(values)]
        lith_series = part[LITHOLOGY_COLUMN]
        lith_unique_count = int(lith_series.dropna().nunique())
        if lith_series.notna().any():
            dominant_lithology = lith_series.value_counts(dropna=True).index[0]
        else:
            dominant_lithology = np.nan
        finite_count = int(len(values))
        unique_count = int(np.unique(values).size) if finite_count else 0
        is_constant = bool(finite_count > 0 and unique_count == 1)

        if finite_count > 0:
            vmin = float(np.min(values))
            vmax = float(np.max(values))
            vmean = float(np.mean(values))
            vstd = float(np.std(values))
        else:
            vmin = np.nan
            vmax = np.nan
            vmean = np.nan
            vstd = np.nan

        fig, ax = plt.subplots(figsize=(7, 5))
        if finite_count > 0:
            if is_constant:
                ax.axvline(values[0], color="#d7301f", linewidth=2.0)
                ax.text(
                    0.02,
                    0.95,
                    f"Constant value: {values[0]:.6g}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                )
            else:
                bins = max(10, int(SURVEY_HISTOGRAM_BINS))
                ax.hist(values, bins=bins, color="#2c7fb8", alpha=0.85)
        else:
            ax.text(
                0.5,
                0.5,
                "No finite values",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )

        ax.set_title(
            f"Survey {project_id} | n={len(part)} | finite={finite_count} | unique={unique_count}"
        )
        ax.set_xlabel(VALUE_COLUMN)
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.2)
        fig.tight_layout()

        plot_name = f"survey_{_safe_name(project_id)}.png"
        plot_path = hist_dir / plot_name
        fig.savefig(plot_path, dpi=130)
        plt.close(fig)

        summary_rows.append(
            {
                "project_id": project_id,
                "sample_count": int(len(part)),
                "lithology_unique_count": lith_unique_count,
                "dominant_lithology": dominant_lithology,
                "finite_count": finite_count,
                "unique_count": unique_count,
                "is_constant": is_constant,
                "min_value": vmin,
                "max_value": vmax,
                "mean_value": vmean,
                "std_value": vstd,
                "histogram_plot": str(plot_path.relative_to(OUTPUT_DIR)),
            }
        )
        progress.update(1)

    progress.close()
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_path = OUTPUT_DIR / SURVEY_HIST_SUMMARY_CSV_NAME
        summary_df.to_csv(summary_path, index=False)
        constant_count = int(summary_df["is_constant"].sum())
        print(f"Saved survey histograms: {hist_dir}")
        print(f"Saved survey histogram summary: {summary_path}")
        print(
            f"Constant-value surveys: {constant_count}/{len(summary_df)} "
            f"({100.0 * constant_count / len(summary_df):.1f}%)"
        )
    return summary_df


def build_project_footprints(gdf: gpd.GeoDataFrame) -> dict:
    footprints = {}
    for project_id, part in gdf.groupby(PROJECT_COLUMN):
        if pd.isna(project_id) or part.empty:
            continue
        footprint = part.geometry.union_all().convex_hull
        if not footprint.is_empty:
            footprints[project_id] = footprint
    return footprints


def pair_values_by_rank(
    ref_points: gpd.GeoDataFrame,
    cand_points: gpd.GeoDataFrame,
    max_pairs: int,
):
    ref_vals = pd.to_numeric(ref_points[VALUE_COLUMN], errors="coerce").to_numpy(
        dtype=float
    )
    cand_vals = pd.to_numeric(cand_points[VALUE_COLUMN], errors="coerce").to_numpy(
        dtype=float
    )
    ref_vals = np.sort(ref_vals[np.isfinite(ref_vals)])
    cand_vals = np.sort(cand_vals[np.isfinite(cand_vals)])

    if len(ref_vals) == 0 or len(cand_vals) == 0:
        return None, None, np.nan

    n = min(len(ref_vals), len(cand_vals), max_pairs if max_pairs else np.inf)
    n = int(n)
    if n <= 0:
        return None, None, np.nan

    q = np.linspace(0.0, 1.0, n)
    ref_ranked = np.quantile(ref_vals, q)
    cand_ranked = np.quantile(cand_vals, q)
    return ref_ranked.astype(float), cand_ranked.astype(float), np.nan


def save_regression_plot(
    plot_path: Path,
    fit_ref: np.ndarray,
    fit_cand: np.ndarray,
    slope: float,
    intercept: float,
    post_slope: float,
    post_intercept: float,
    project_id: object,
    reference_project_id: object,
    pairing_mode: str,
) -> None:
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        fit_cand,
        fit_ref,
        s=12,
        alpha=0.35,
        color="#2c7fb8",
        label="trimmed pairs (pre)",
    )
    corrected_cand = slope * fit_cand + intercept
    ax.scatter(
        corrected_cand,
        fit_ref,
        s=12,
        alpha=0.55,
        color="#1a9850",
        label="trimmed pairs (post)",
    )

    x_min = float(np.nanmin(np.concatenate([fit_cand, corrected_cand, fit_ref])))
    x_max = float(np.nanmax(np.concatenate([fit_cand, corrected_cand, fit_ref])))
    x_lo = x_min
    x_hi = x_max
    if np.isfinite(x_lo) and np.isfinite(x_hi) and x_hi > x_lo:
        x_line = np.linspace(x_lo, x_hi, 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="#d7301f", linewidth=1.8, label="pre fit")
        ax.plot(
            x_line, x_line, color="black", linewidth=1.2, linestyle="--", label="1:1"
        )

    ax.set_title(
        f"Ref={reference_project_id} | Cand={project_id}\n"
        f"mode={pairing_mode} | pre=({slope:.4f}, {intercept:.4f}) | "
        f"post=({post_slope:.4f}, {post_intercept:.4f})"
    )
    ax.set_xlabel("Candidate values")
    ax.set_ylabel("Reference values")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def build_regression_pairs(
    ref_points: gpd.GeoDataFrame,
    cand_points: gpd.GeoDataFrame,
    rng: np.random.Generator,
    spatial_mode: str,
):
    del rng, spatial_mode  # Pairing is now rank-based and lithology-agnostic.
    ref_pairs, cand_pairs, mean_distance = pair_values_by_rank(
        ref_points,
        cand_points,
        max_pairs=MAX_PAIR_COUNT,
    )
    if ref_pairs is None or len(ref_pairs) < MIN_TOTAL_PAIR_COUNT:
        return None, None, "insufficient_pairs", np.nan
    return ref_pairs, cand_pairs, "global_rank", mean_distance


def trim_pairs_for_regression(
    ref_values: np.ndarray,
    cand_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    paired = np.isfinite(ref_values) & np.isfinite(cand_values)
    ref_values = ref_values[paired]
    cand_values = cand_values[paired]
    if len(ref_values) == 0:
        return ref_values, cand_values

    ref_lo, ref_hi = np.quantile(
        ref_values, [REGRESSION_LOWER_QUANTILE, REGRESSION_UPPER_QUANTILE]
    )
    cand_lo, cand_hi = np.quantile(
        cand_values, [REGRESSION_LOWER_QUANTILE, REGRESSION_UPPER_QUANTILE]
    )
    keep = (
        (ref_values >= ref_lo)
        & (ref_values <= ref_hi)
        & (cand_values >= cand_lo)
        & (cand_values <= cand_hi)
    )
    return ref_values[keep], cand_values[keep]


def fit_correction(
    ref_values: np.ndarray,
    cand_values: np.ndarray,
) -> tuple[float, float, float, float]:
    slope, intercept = stable_linear_fit(cand_values, ref_values)
    adjusted = slope * cand_values + intercept
    post_slope, post_intercept = stable_linear_fit(adjusted, ref_values)
    return float(slope), float(intercept), float(post_slope), float(post_intercept)


def stable_linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    paired = np.isfinite(x) & np.isfinite(y)
    x = x[paired]
    y = y[paired]

    if len(x) < 2:
        return 1.0, float(np.mean(y) - np.mean(x))

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_centered = x - x_mean
    var_x = float(np.mean(x_centered * x_centered))

    # If candidate values are almost constant, slope estimation is unstable.
    # In that case, apply additive-only correction.
    if var_x <= LINEAR_FIT_MIN_VARIANCE:
        return 1.0, y_mean - x_mean

    cov_xy = float(np.mean(x_centered * (y - y_mean)))
    slope = cov_xy / var_x
    intercept = y_mean - slope * x_mean
    return slope, intercept


def is_low_confidence_slope(slope: float) -> bool:
    if not np.isfinite(slope):
        return True
    return (slope < LOW_CONF_SLOPE_MIN) or (slope > LOW_CONF_SLOPE_MAX)


def apply_correction(
    gdf: gpd.GeoDataFrame,
    project_id,
    slope: float,
    intercept: float,
    step_number: int,
) -> None:
    mask = gdf[PROJECT_COLUMN] == project_id
    gdf.loc[mask, VALUE_COLUMN] = gdf.loc[mask, VALUE_COLUMN] * slope + intercept
    gdf.loc[mask, "LVL_MUL"] = gdf.loc[mask, "LVL_MUL"] * slope
    gdf.loc[mask, "LVL_ADD"] = gdf.loc[mask, "LVL_ADD"] * slope + intercept
    gdf.loc[mask, "LVL_STP"] = step_number


def summarize_metrics(
    log_df: pd.DataFrame,
    spatial_mode: str,
    metric_scope: str = "all_steps",
) -> dict:
    if log_df.empty:
        return {
            "spatial_mode": spatial_mode,
            "metric_scope": metric_scope,
            "steps": 0,
            "mean_applied_slope": np.nan,
            "mean_applied_intercept": np.nan,
            "mean_abs_slope_error": np.nan,
            "mean_abs_intercept": np.nan,
        }

    return {
        "spatial_mode": spatial_mode,
        "metric_scope": metric_scope,
        "steps": int(len(log_df)),
        "mean_applied_slope": float(log_df["slope_applied"].mean()),
        "mean_applied_intercept": float(log_df["intercept_applied"].mean()),
        "mean_abs_slope_error": float((log_df["slope_applied"] - 1.0).abs().mean()),
        "mean_abs_intercept": float(log_df["intercept_applied"].abs().mean()),
    }


def summarize_common_project_metrics(run_logs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not run_logs:
        return pd.DataFrame()

    project_sets = [set(df["project_id"]) for df in run_logs.values()]
    common_projects = set.intersection(*project_sets) if project_sets else set()

    rows = []
    for mode, log_df in run_logs.items():
        common_df = log_df[log_df["project_id"].isin(common_projects)].copy()
        metrics = summarize_metrics(
            common_df,
            spatial_mode=mode,
            metric_scope="common_projects_only",
        )
        mode_steps = int(len(log_df))
        common_steps = int(len(common_df))
        metrics["common_projects"] = int(len(common_projects))
        metrics["mode_total_steps"] = mode_steps
        metrics["common_step_share"] = (
            float(common_steps / mode_steps) if mode_steps > 0 else np.nan
        )
        rows.append(metrics)

    return pd.DataFrame(rows)


def is_fully_contained(inner_geom, outer_geom) -> bool:
    if inner_geom is None or outer_geom is None:
        return False
    if inner_geom.is_empty or outer_geom.is_empty:
        return False
    try:
        return bool(outer_geom.covers(inner_geom))
    except Exception:
        return bool(inner_geom.within(outer_geom))


def run_leveling(
    gdf: gpd.GeoDataFrame,
    spatial_mode: str,
    final_output_name: str,
    log_csv_name: str,
) -> tuple[dict, pd.DataFrame]:
    counts = project_counts(gdf)
    if counts.empty:
        raise ValueError("No valid project IDs found.")

    ordered_projects = list(counts.index)
    footprints = build_project_footprints(gdf)
    # GroupBy.indices returns positional indices, so we must use iloc (not loc).
    project_indices = gdf.groupby(PROJECT_COLUMN, sort=False).indices
    rng = np.random.default_rng(RANDOM_SEED)

    gdf["LVL_MUL"] = 1.0
    gdf["LVL_ADD"] = 0.0
    gdf["LVL_STP"] = 0
    gdf[ISLAND_COLUMN] = 0
    gdf[LOW_CONF_COLUMN] = 0

    project_island: dict[object, int] = {}
    next_island_id = 1
    corrected_projects: set = set()
    excluded_low_unique_projects: set = set()
    excluded_insufficient_pairs_projects: set = set()
    log_rows = []
    step_number = 0
    skipped_low_conf_count = 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_dir = OUTPUT_DIR / REGRESSION_PLOTS_DIRNAME
    if SAVE_REGRESSION_PLOTS and plt is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_REGRESSION_PLOTS and plt is None:
        print("Regression plot saving requested, but matplotlib is not available.")

    print(
        f"Reference project (largest): {ordered_projects[0]} ({counts.iloc[0]} samples)"
    )
    print("\nPhase 1: containment-only leveling (fully nested surveys)")
    print("Rules: no lithology split, rank-based pairing only, no phase 2.")

    reference_progress = make_progress(
        len(ordered_projects),
        f"Containment sweep ({spatial_mode})",
        unit="reference",
    )

    for reference_project in ordered_projects:
        if reference_project in excluded_low_unique_projects:
            reference_progress.set_postfix_str(
                f"skipped {reference_project} (low-unique after trim)"
            )
            reference_progress.update(1)
            continue
        if reference_project in excluded_insufficient_pairs_projects:
            reference_progress.set_postfix_str(
                f"skipped {reference_project} (insufficient pairs)"
            )
            reference_progress.update(1)
            continue
        if reference_project not in project_indices:
            reference_progress.set_postfix_str(
                f"skipped {reference_project} (no points)"
            )
            reference_progress.update(1)
            continue

        if reference_project not in project_island:
            project_island[reference_project] = next_island_id
            next_island_id += 1
        ref_island = project_island[reference_project]
        gdf.loc[gdf[PROJECT_COLUMN] == reference_project, ISLAND_COLUMN] = ref_island

        ref_footprint = footprints.get(reference_project)
        if ref_footprint is None or ref_footprint.is_empty:
            reference_progress.set_postfix_str(
                f"skipped {reference_project} (missing footprint)"
            )
            reference_progress.update(1)
            continue

        candidates = []
        for candidate in ordered_projects:
            if (
                candidate == reference_project
                or candidate in corrected_projects
                or candidate in excluded_low_unique_projects
                or candidate in excluded_insufficient_pairs_projects
            ):
                continue
            if counts.get(candidate, 0) >= counts.get(reference_project, 0):
                continue
            cand_footprint = footprints.get(candidate)
            if cand_footprint is None or cand_footprint.is_empty:
                continue
            if is_fully_contained(cand_footprint, ref_footprint):
                candidates.append(candidate)

        if candidates:
            print(
                f"Reference {reference_project} (island={ref_island}) -> "
                f"{len(candidates)} fully contained candidate(s)."
            )

        ref_points = gdf.iloc[project_indices[reference_project]]
        for candidate in candidates:
            cand_points = gdf.iloc[project_indices[candidate]]
            ref_pairs, cand_pairs, pairing_mode, avg_pair_distance = (
                build_regression_pairs(
                    ref_points,
                    cand_points,
                    rng,
                    spatial_mode=spatial_mode,
                )
            )
            if ref_pairs is None:
                excluded_insufficient_pairs_projects.add(candidate)
                print(f"Skipped {candidate}: insufficient raw pairs.")
                continue

            raw_pair_count = int(len(ref_pairs))
            fit_ref, fit_cand = trim_pairs_for_regression(ref_pairs, cand_pairs)
            fit_pair_count = int(len(fit_ref))
            if fit_pair_count < MIN_TRIMMED_PAIR_COUNT:
                excluded_insufficient_pairs_projects.add(candidate)
                print(
                    f"Skipped {candidate}: insufficient trimmed pairs "
                    f"({fit_pair_count}/{raw_pair_count})."
                )
                continue
            fit_ref_unique = int(np.unique(fit_ref).size)
            fit_cand_unique = int(np.unique(fit_cand).size)
            if (
                fit_ref_unique < MIN_UNIQUE_VALUES_PER_SURVEY
                or fit_cand_unique < MIN_UNIQUE_VALUES_PER_SURVEY
            ):
                excluded_low_unique_projects.add(candidate)
                print(
                    f"Skipped {candidate}: low unique after trim "
                    f"(ref_unique={fit_ref_unique}, cand_unique={fit_cand_unique}, "
                    f"pairs={fit_pair_count}/{raw_pair_count})"
                )
                continue

            slope, intercept, post_slope, post_intercept = fit_correction(
                fit_ref, fit_cand
            )
            low_confidence_flag = is_low_confidence_slope(slope)
            if low_confidence_flag:
                skipped_low_conf_count += 1
                gdf.loc[gdf[PROJECT_COLUMN] == candidate, LOW_CONF_COLUMN] = 1
                continue

            step_number += 1
            apply_correction(gdf, candidate, slope, intercept, step_number)
            corrected_projects.add(candidate)
            project_island[candidate] = ref_island
            gdf.loc[gdf[PROJECT_COLUMN] == candidate, ISLAND_COLUMN] = ref_island

            plot_relpath = ""
            if SAVE_REGRESSION_PLOTS and plt is not None:
                plot_name = (
                    f"step_{step_number:04d}"
                    f"__ref_{_safe_name(reference_project)}"
                    f"__cand_{_safe_name(candidate)}.png"
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
                    project_id=candidate,
                    reference_project_id=reference_project,
                    pairing_mode=pairing_mode,
                )
                plot_relpath = str(plot_path.relative_to(OUTPUT_DIR))

            cand_footprint = footprints.get(candidate)
            ref_overlap_count = int(
                ref_points.geometry.intersects(cand_footprint).sum()
            )
            cand_overlap_count = int(len(cand_points))

            log_rows.append(
                {
                    "step": step_number,
                    "phase": "containment",
                    "component_id": ref_island,
                    "project_id": candidate,
                    "reference_project_id": reference_project,
                    "samples_in_project": int(counts.get(candidate, 0)),
                    "overlap_ref_points": ref_overlap_count,
                    "overlap_cand_points": cand_overlap_count,
                    "pair_count": fit_pair_count,
                    "pair_count_raw": raw_pair_count,
                    "slope_applied": slope,
                    "intercept_applied": intercept,
                    "post_slope": post_slope,
                    "post_intercept": post_intercept,
                    "pairing_mode": pairing_mode,
                    "spatial_mode": spatial_mode,
                    "avg_pair_distance": avg_pair_distance,
                    "shared_boundary_length_m": np.nan,
                    "boundary_corridor_width_m": np.nan,
                    "reference_component_id": ref_island,
                    "low_confidence_flag": low_confidence_flag,
                    "regression_plot": plot_relpath,
                }
            )

            print(
                f"Step {step_number}: leveled {candidate} "
                f"(phase=containment, component={ref_island}, ref={reference_project}, "
                f"mode={pairing_mode}, pairs={fit_pair_count}/{raw_pair_count} fit/raw, "
                f"applied slope={slope:.4f}, intercept={intercept:.4f})"
            )

        reference_progress.set_postfix_str(
            f"ref={reference_project} | corrected={len(corrected_projects)}"
        )
        reference_progress.update(1)

    reference_progress.close()

    total_islands = len(set(project_island.values()))
    untouched_count = int(len(ordered_projects) - len(corrected_projects))
    print(
        f"Phase 1 complete: corrected {len(corrected_projects)} survey(s), "
        f"untouched/root surveys {untouched_count}, islands={total_islands}, "
        f"excluded_low_unique={len(excluded_low_unique_projects)}, "
        f"excluded_insufficient_pairs={len(excluded_insufficient_pairs_projects)}."
    )
    print("Phase 2 skipped entirely by configuration.")

    final_out = gdf.copy()
    excluded_reasons: dict[object, str] = {}
    for project_id in excluded_low_unique_projects:
        excluded_reasons[project_id] = "low_unique_after_trim"
    for project_id in excluded_insufficient_pairs_projects:
        excluded_reasons[project_id] = "insufficient_pairs"

    excluded_projects = set(excluded_reasons)
    if excluded_projects:
        excluded_mask = final_out[PROJECT_COLUMN].isin(excluded_projects)
        removed_excluded_rows = int(excluded_mask.sum())
        final_out = final_out[~excluded_mask].copy()
        low_unique_path = OUTPUT_DIR / LOW_UNIQUE_SURVEYS_CSV_NAME
        pd.DataFrame(
            {
                PROJECT_COLUMN: sorted(excluded_projects),
                "reason": [excluded_reasons[p] for p in sorted(excluded_projects)],
            }
        ).to_csv(low_unique_path, index=False)
        print(
            f"Excluded {len(excluded_projects)} survey(s) "
            f"({removed_excluded_rows} rows removed): "
            f"{len(excluded_low_unique_projects)} low-unique, "
            f"{len(excluded_insufficient_pairs_projects)} insufficient-pairs."
        )
        print(f"Saved excluded-survey list: {low_unique_path}")
    neg_out_mask = final_out[VALUE_COLUMN] < 0
    dropped_negative_rows = int(neg_out_mask.sum())
    if dropped_negative_rows:
        final_out = final_out[~neg_out_mask].copy()
        print(
            f"Removed {dropped_negative_rows} rows with negative leveled "
            f"{VALUE_COLUMN} from final output."
        )

    if not final_out.empty:
        upper_threshold = float(
            final_out[VALUE_COLUMN].quantile(OUTPUT_UPPER_PERCENTILE)
        )
        high_out_mask = final_out[VALUE_COLUMN] > upper_threshold
        dropped_high_rows = int(high_out_mask.sum())
        if dropped_high_rows:
            final_out = final_out[~high_out_mask].copy()
            print(
                f"Removed {dropped_high_rows} rows above the "
                f"{int(OUTPUT_UPPER_PERCENTILE * 100)}th percentile "
                f"({VALUE_COLUMN} > {upper_threshold:.6g}) from final output."
            )

    final_out_path = OUTPUT_DIR / final_output_name
    final_out.to_file(final_out_path)
    print(f"Saved final leveled shapefile: {final_out_path}")

    log_df = pd.DataFrame(log_rows)
    if log_rows:
        log_path = OUTPUT_DIR / log_csv_name
        log_df.to_csv(log_path, index=False)
        print(f"Saved leveling log: {log_path}")

    print(f"Skipped low-confidence corrections: {skipped_low_conf_count}")
    logged_low_conf = (
        int(log_df["low_confidence_flag"].fillna(False).sum())
        if ("low_confidence_flag" in log_df.columns and not log_df.empty)
        else 0
    )
    total_low_conf = skipped_low_conf_count + logged_low_conf
    total_fit_attempts = len(log_df) + skipped_low_conf_count
    if total_fit_attempts > 0:
        low_conf_pct = 100.0 * total_low_conf / total_fit_attempts
        print(
            f"Low-confidence fit attempts: {total_low_conf}/{total_fit_attempts} "
            f"({low_conf_pct:.1f}%)"
        )

    metrics = summarize_metrics(
        log_df,
        spatial_mode=spatial_mode,
        metric_scope="all_steps",
    )
    print(
        f"Metrics [{spatial_mode}]: "
        f"mean_slope={metrics['mean_applied_slope']:.4f}, "
        f"mean_intercept={metrics['mean_applied_intercept']:.4f}, "
        f"mean|slope-1|={metrics['mean_abs_slope_error']:.4f}, "
        f"mean|intercept|={metrics['mean_abs_intercept']:.4f}"
    )
    return metrics, log_df


def main() -> None:
    print(f"Reading shapefile: {INPUT_SHP}")
    gdf = gpd.read_file(INPUT_SHP)

    print(f"Rows: {len(gdf)}")
    print("Columns:", ", ".join(map(str, gdf.columns)))

    validate_required_columns(gdf)
    unknown_count = normalize_project_ids(gdf)
    if unknown_count:
        print(
            f"Normalized missing/None project IDs to 'Unknown': {unknown_count} rows."
        )

    imputed_count, nan_count = impute_bdl_values(gdf)
    print(
        f"Imputed {imputed_count} negative {VALUE_COLUMN} values "
        f"as half absolute value (below-detection handling)."
    )
    if nan_count:
        print(f"Warning: {nan_count} {VALUE_COLUMN} values are non-numeric/NaN.")

    counts = project_counts(gdf)
    print("\nTop projects by sample count:")
    print(counts.head(10))

    print(
        f"Low-unique filtering is applied after pair trimming "
        f"(threshold={MIN_UNIQUE_VALUES_PER_SURVEY} unique values)."
    )

    if SAVE_SURVEY_HISTOGRAMS:
        export_survey_histograms(gdf)

    if RUN_DISTANCE_COMPARISON:
        overall_rows = []
        run_logs = {}
        for mode in DISTANCE_COMPARISON_MODES:
            print(f"\n--- Running leveling with {mode} ---")
            metrics, log_df = run_leveling(
                gdf.copy(),
                spatial_mode=mode,
                final_output_name=f"leveled_all_{mode}.shp",
                log_csv_name=f"leveling_log_{mode}.csv",
            )
            overall_rows.append(metrics)
            run_logs[mode] = log_df

        overall_df = pd.DataFrame(overall_rows)
        common_df = summarize_common_project_metrics(run_logs)
        if common_df.empty:
            comparison_df = overall_df.copy()
        else:
            comparison_df = common_df.copy()
        comparison_path = OUTPUT_DIR / COMPARISON_METRICS_CSV_NAME
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nSaved comparison metrics: {comparison_path}")
        print("\nOverall metrics (all leveled steps):")
        print(overall_df.to_string(index=False))
        if not common_df.empty:
            print("\nApples-to-apples metrics (common projects only):")
        print(comparison_df.to_string(index=False))
    else:
        run_leveling(
            gdf,
            spatial_mode=SPATIAL_PAIRING_MODE,
            final_output_name=FINAL_OUTPUT_NAME,
            log_csv_name=LOG_CSV_NAME,
        )


if __name__ == "__main__":
    main()
