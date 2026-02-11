from __future__ import annotations

import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from level_core import (
    apply_correction,
    build_project_counts,
    build_project_footprints,
    build_survey_levelability,
    coerce_numeric_columns,
    fit_correction,
    is_fully_overlapped,
    make_progress,
    normalize_project_ids,
    pair_values_by_rank,
    safe_name,
    save_regression_plot,
    stable_linear_fit,
    trim_pairs,
    validate_required_columns,
)

# -------------------------- User-editable settings --------------------------
INPUT_SHP = Path("output/leveled_full_overlap.shp")
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
MIN_PAIR_COUNT = 20
MAX_PAIR_COUNT = 1000
MIN_UNIQUE_VALUES = 3
TRIM_LOWER_QUANTILE = 0.1
TRIM_UPPER_QUANTILE = 0.9
LINEAR_FIT_MIN_VARIANCE = 1e-16

# Phase-2 cascade parameters (partial leveling)
BUFFER_DISTANCE_KM = 20.0
PARTIAL_LEVELING_RATE = 0.3
MAX_CYCLES = 50
MIN_CYCLE_UPDATES = 1
RANDOMIZE_CANDIDATE_ROUTE_PER_CYCLE = True
RANDOM_SEED = 42
SKIP_FULL_OVERLAP_PAIRS = True

# Outputs
SAVE_REGRESSION_PLOTS = True
CLEAR_PLOT_DIR_ON_START = True
REGRESSION_PLOTS_DIRNAME = "phase2_regression_plots"
PHASE2_QA_CSV_NAME = "phase2_survey_levelability.csv"
PHASE2_LOG_CSV_NAME = "phase2_leveling_log.csv"
PHASE2_EXCLUDED_CSV_NAME = "phase2_excluded_surveys.csv"
PHASE2_OUTPUT_NAME = "phase2_leveled_partial_overlap.shp"
KEEP_ALL_PHASE2_INPUT_SURVEYS_IN_OUTPUT = True

# Simple output stabilizer for phase 2
CLIP_PHASE2_VALUES = True
PHASE2_MIN_VALUE = 0.0
PHASE2_MAX_PERCENTILE = 99.0
# ---------------------------------------------------------------------------


def _to_metric_crs(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, str]:
    if gdf.crs is None:
        raise ValueError(
            "Input GeoDataFrame has no CRS. A valid CRS is required for 20 km buffering."
        )

    if getattr(gdf.crs, "is_geographic", False):
        metric_crs = gdf.estimate_utm_crs()
        if metric_crs is None:
            metric_crs = "EPSG:3857"
        gdf_metric = gdf.to_crs(metric_crs)
        return gdf_metric, str(metric_crs)

    return gdf, str(gdf.crs)


def _init_phase2_columns(gdf: gpd.GeoDataFrame) -> None:
    if "LVL_MUL" not in gdf.columns:
        gdf["LVL_MUL"] = 1.0
    if "LVL_ADD" not in gdf.columns:
        gdf["LVL_ADD"] = 0.0
    if "LVL_STP" not in gdf.columns:
        gdf["LVL_STP"] = 0
    if "LVL_REF" not in gdf.columns:
        gdf["LVL_REF"] = ""
    if "LVL_STAT" not in gdf.columns:
        gdf["LVL_STAT"] = "eligible_unleveled"

    gdf["P2_MUL"] = 1.0
    gdf["P2_ADD"] = 0.0
    gdf["P2_STP"] = 0
    gdf["P2_CYC"] = 0
    gdf["P2_REF"] = ""
    gdf["P2_STAT"] = "eligible_unleveled"


def _apply_phase2_correction(
    gdf: gpd.GeoDataFrame,
    project_id: object,
    slope: float,
    intercept: float,
    step: int,
    cycle: int,
    reference_project: object,
) -> None:
    # Reuse phase-1 updater for value + cumulative LVL_* tracking.
    apply_correction(
        gdf,
        project_column=PROJECT_COLUMN,
        value_column=IMPUTED_VALUE_COLUMN,
        project_id=project_id,
        slope=slope,
        intercept=intercept,
        step=step,
        reference_project=reference_project,
    )

    mask = gdf[PROJECT_COLUMN] == project_id
    gdf.loc[mask, "P2_MUL"] = gdf.loc[mask, "P2_MUL"] * slope
    gdf.loc[mask, "P2_ADD"] = gdf.loc[mask, "P2_ADD"] * slope + intercept
    gdf.loc[mask, "P2_STP"] = step
    gdf.loc[mask, "P2_CYC"] = cycle
    gdf.loc[mask, "P2_REF"] = str(reference_project)
    gdf.loc[mask, "P2_STAT"] = "leveled_partial"


def _clip_project_values(
    gdf: gpd.GeoDataFrame,
    project_id: object,
    min_value: float,
    max_value: float,
) -> None:
    mask = gdf[PROJECT_COLUMN] == project_id
    gdf.loc[mask, IMPUTED_VALUE_COLUMN] = gdf.loc[mask, IMPUTED_VALUE_COLUMN].clip(
        lower=min_value,
        upper=max_value,
    )


def run_phase2_leveling(gdf: gpd.GeoDataFrame, survey_qa: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    levelable_projects = set(survey_qa.loc[survey_qa["levelable"], PROJECT_COLUMN])
    excluded_quality_projects = set(
        survey_qa.loc[~survey_qa["levelable"], PROJECT_COLUMN]
    )

    counts = build_project_counts(gdf, PROJECT_COLUMN, levelable_projects)
    if counts.empty:
        raise ValueError("No surveys passed the levelability rules for phase 2.")

    clip_max_value = np.nan
    if CLIP_PHASE2_VALUES:
        all_vals = pd.to_numeric(gdf[IMPUTED_VALUE_COLUMN], errors="coerce").to_numpy(
            dtype=float
        )
        finite_vals = all_vals[np.isfinite(all_vals)]
        if len(finite_vals) == 0:
            raise ValueError(
                f"No finite values in {IMPUTED_VALUE_COLUMN}; cannot compute clip percentile."
            )
        clip_max_value = float(
            np.nanquantile(finite_vals, PHASE2_MAX_PERCENTILE / 100.0)
        )

    ordered_projects = list(counts.index)
    base_project = ordered_projects[0]

    gdf_metric, metric_crs_name = _to_metric_crs(gdf)
    footprints = build_project_footprints(
        gdf_metric, PROJECT_COLUMN, levelable_projects
    )
    buffer_m = float(BUFFER_DISTANCE_KM) * 1000.0
    buffered_footprints = {
        pid: fp.buffer(buffer_m) for pid, fp in footprints.items() if not fp.is_empty
    }

    project_indices = gdf.groupby(PROJECT_COLUMN, sort=False).indices
    project_positions: dict[str, np.ndarray] = {
        pid: np.asarray(pos, dtype=int) for pid, pos in project_indices.items()
    }
    project_geoms_metric: dict[str, gpd.GeoSeries] = {
        pid: gdf_metric.geometry.iloc[pos].reset_index(drop=True)
        for pid, pos in project_positions.items()
    }

    _init_phase2_columns(gdf)
    if excluded_quality_projects:
        gdf.loc[gdf[PROJECT_COLUMN].isin(excluded_quality_projects), "P2_STAT"] = (
            "excluded_quality"
        )

    plot_dir = OUTPUT_DIR / REGRESSION_PLOTS_DIRNAME
    if SAVE_REGRESSION_PLOTS:
        if CLEAR_PLOT_DIR_ON_START and plot_dir.exists():
            shutil.rmtree(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

    excluded_low_unique_values: set[str] = set()
    log_rows: list[dict] = []
    rng = np.random.default_rng(RANDOM_SEED)

    step = 0
    total_ticks = MAX_CYCLES * len(ordered_projects)
    progress = make_progress(
        total_ticks, "Phase 2 partial-overlap cascade", unit="survey"
    )

    print(
        f"Phase 2 setup: base_project={base_project}, buffer={BUFFER_DISTANCE_KM:.1f} km, "
        f"leveling_rate={PARTIAL_LEVELING_RATE:.3f}, max_cycles={MAX_CYCLES}, "
        f"candidate_route_randomized={RANDOMIZE_CANDIDATE_ROUTE_PER_CYCLE}, "
        f"clip_values={CLIP_PHASE2_VALUES}, clip_range=[{PHASE2_MIN_VALUE:.3g}, {clip_max_value:.3g}], "
        f"metric_crs={metric_crs_name}"
    )

    completed_cycles = 0
    for cycle in range(1, MAX_CYCLES + 1):
        cycle_updates = 0
        refs = ordered_projects

        for reference_project in refs:
            progress.update(1)

            if reference_project in excluded_low_unique_values:
                continue
            if reference_project not in footprints:
                continue
            if reference_project not in buffered_footprints:
                continue
            if reference_project not in project_positions:
                continue

            ref_pos = project_positions[reference_project]
            ref_geom = project_geoms_metric[reference_project]
            ref_vals_full = pd.to_numeric(
                gdf.iloc[ref_pos][IMPUTED_VALUE_COLUMN], errors="coerce"
            ).to_numpy(dtype=float)
            ref_unique_count = int(
                np.unique(ref_vals_full[np.isfinite(ref_vals_full)]).size
            )
            if ref_unique_count < MIN_UNIQUE_VALUES:
                excluded_low_unique_values.add(reference_project)
                gdf.loc[gdf[PROJECT_COLUMN] == reference_project, "P2_STAT"] = (
                    "excluded_low_unique_values"
                )
                log_rows.append(
                    {
                        "cycle": cycle,
                        "step": np.nan,
                        "status": "excluded_low_unique_values",
                        "reference_project": reference_project,
                        "candidate_project": np.nan,
                        "reference_unique_count": ref_unique_count,
                        "candidate_unique_count": np.nan,
                        "overlap_ref_points": 0,
                        "overlap_cand_points": 0,
                        "pair_count_raw": 0,
                        "pair_count_trimmed": 0,
                        "slope_full": np.nan,
                        "intercept_full": np.nan,
                        "leveling_rate": PARTIAL_LEVELING_RATE,
                        "slope_applied": np.nan,
                        "intercept_applied": np.nan,
                        "post_slope": np.nan,
                        "post_intercept": np.nan,
                        "regression_plot": "",
                    }
                )
                continue

            ref_buffer = buffered_footprints[reference_project]
            candidate_rows: list[tuple[str, int, np.ndarray, np.ndarray]] = []
            for candidate in ordered_projects:
                if candidate == reference_project:
                    continue
                if candidate == base_project:
                    continue
                if candidate in excluded_low_unique_values:
                    continue
                if candidate not in footprints:
                    continue
                if candidate not in buffered_footprints:
                    continue
                if candidate not in project_positions:
                    continue

                if SKIP_FULL_OVERLAP_PAIRS and (
                    is_fully_overlapped(
                        footprints[candidate], footprints[reference_project]
                    )
                    or is_fully_overlapped(
                        footprints[reference_project], footprints[candidate]
                    )
                ):
                    continue

                if not ref_buffer.intersects(footprints[candidate]):
                    continue

                cand_buffer = buffered_footprints[candidate]
                ref_mask = ref_geom.within(cand_buffer).to_numpy(dtype=bool)
                if not np.any(ref_mask):
                    continue

                cand_geom = project_geoms_metric[candidate]
                cand_mask = cand_geom.within(ref_buffer).to_numpy(dtype=bool)
                if not np.any(cand_mask):
                    continue

                overlap_score = int(np.sum(ref_mask) + np.sum(cand_mask))
                candidate_rows.append((candidate, overlap_score, ref_mask, cand_mask))

            if RANDOMIZE_CANDIDATE_ROUTE_PER_CYCLE:
                rng.shuffle(candidate_rows)
            else:
                candidate_rows.sort(
                    key=lambda item: (
                        -item[1],
                        -int(counts.get(item[0], 0)),
                        str(item[0]),
                    )
                )

            for candidate, _, ref_mask, cand_mask in candidate_rows:
                cand_pos = project_positions[candidate]
                ref_overlap_pos = ref_pos[ref_mask]
                cand_overlap_pos = cand_pos[cand_mask]

                ref_vals = pd.to_numeric(
                    gdf.iloc[ref_overlap_pos][IMPUTED_VALUE_COLUMN], errors="coerce"
                ).to_numpy(dtype=float)
                cand_vals = pd.to_numeric(
                    gdf.iloc[cand_overlap_pos][IMPUTED_VALUE_COLUMN], errors="coerce"
                ).to_numpy(dtype=float)

                ref_pairs, cand_pairs = pair_values_by_rank(
                    ref_vals,
                    cand_vals,
                    min_pair_count=MIN_PAIR_COUNT,
                    max_pair_count=MAX_PAIR_COUNT,
                )

                if ref_pairs is None:
                    log_rows.append(
                        {
                            "cycle": cycle,
                            "step": np.nan,
                            "status": "skipped_insufficient_pairs",
                            "reference_project": reference_project,
                            "candidate_project": candidate,
                            "reference_unique_count": np.nan,
                            "candidate_unique_count": np.nan,
                            "overlap_ref_points": int(len(ref_overlap_pos)),
                            "overlap_cand_points": int(len(cand_overlap_pos)),
                            "pair_count_raw": 0,
                            "pair_count_trimmed": 0,
                            "slope_full": np.nan,
                            "intercept_full": np.nan,
                            "leveling_rate": PARTIAL_LEVELING_RATE,
                            "slope_applied": np.nan,
                            "intercept_applied": np.nan,
                            "post_slope": np.nan,
                            "post_intercept": np.nan,
                            "regression_plot": "",
                        }
                    )
                    continue

                raw_count = int(len(ref_pairs))
                fit_ref, fit_cand = trim_pairs(
                    ref_pairs,
                    cand_pairs,
                    trim_lower_quantile=TRIM_LOWER_QUANTILE,
                    trim_upper_quantile=TRIM_UPPER_QUANTILE,
                )
                fit_count = int(len(fit_ref))
                ref_trimmed_unique_count = int(
                    np.unique(fit_ref[np.isfinite(fit_ref)]).size
                )
                cand_trimmed_unique_count = int(
                    np.unique(fit_cand[np.isfinite(fit_cand)]).size
                )

                if fit_count < MIN_PAIR_COUNT:
                    log_rows.append(
                        {
                            "cycle": cycle,
                            "step": np.nan,
                            "status": "skipped_insufficient_pairs",
                            "reference_project": reference_project,
                            "candidate_project": candidate,
                            "reference_unique_count": ref_trimmed_unique_count,
                            "candidate_unique_count": cand_trimmed_unique_count,
                            "overlap_ref_points": int(len(ref_overlap_pos)),
                            "overlap_cand_points": int(len(cand_overlap_pos)),
                            "pair_count_raw": raw_count,
                            "pair_count_trimmed": fit_count,
                            "slope_full": np.nan,
                            "intercept_full": np.nan,
                            "leveling_rate": PARTIAL_LEVELING_RATE,
                            "slope_applied": np.nan,
                            "intercept_applied": np.nan,
                            "post_slope": np.nan,
                            "post_intercept": np.nan,
                            "regression_plot": "",
                        }
                    )
                    continue

                if ref_trimmed_unique_count < MIN_UNIQUE_VALUES:
                    log_rows.append(
                        {
                            "cycle": cycle,
                            "step": np.nan,
                            "status": "skipped_reference_low_unique_trim",
                            "reference_project": reference_project,
                            "candidate_project": candidate,
                            "reference_unique_count": ref_trimmed_unique_count,
                            "candidate_unique_count": cand_trimmed_unique_count,
                            "overlap_ref_points": int(len(ref_overlap_pos)),
                            "overlap_cand_points": int(len(cand_overlap_pos)),
                            "pair_count_raw": raw_count,
                            "pair_count_trimmed": fit_count,
                            "slope_full": np.nan,
                            "intercept_full": np.nan,
                            "leveling_rate": PARTIAL_LEVELING_RATE,
                            "slope_applied": np.nan,
                            "intercept_applied": np.nan,
                            "post_slope": np.nan,
                            "post_intercept": np.nan,
                            "regression_plot": "",
                        }
                    )
                    continue

                if cand_trimmed_unique_count < MIN_UNIQUE_VALUES:
                    excluded_low_unique_values.add(candidate)
                    gdf.loc[gdf[PROJECT_COLUMN] == candidate, "P2_STAT"] = (
                        "excluded_low_unique_values"
                    )
                    log_rows.append(
                        {
                            "cycle": cycle,
                            "step": np.nan,
                            "status": "excluded_low_unique_values",
                            "reference_project": reference_project,
                            "candidate_project": candidate,
                            "reference_unique_count": ref_trimmed_unique_count,
                            "candidate_unique_count": cand_trimmed_unique_count,
                            "overlap_ref_points": int(len(ref_overlap_pos)),
                            "overlap_cand_points": int(len(cand_overlap_pos)),
                            "pair_count_raw": raw_count,
                            "pair_count_trimmed": fit_count,
                            "slope_full": np.nan,
                            "intercept_full": np.nan,
                            "leveling_rate": PARTIAL_LEVELING_RATE,
                            "slope_applied": np.nan,
                            "intercept_applied": np.nan,
                            "post_slope": np.nan,
                            "post_intercept": np.nan,
                            "regression_plot": "",
                        }
                    )
                    continue

                slope_full, intercept_full, _, _ = fit_correction(
                    fit_ref,
                    fit_cand,
                    min_variance=LINEAR_FIT_MIN_VARIANCE,
                )
                slope_applied = (slope_full - 1.0) * PARTIAL_LEVELING_RATE + 1.0
                intercept_applied = intercept_full * PARTIAL_LEVELING_RATE

                corrected_preview = slope_applied * fit_cand + intercept_applied
                post_slope, post_intercept = stable_linear_fit(
                    corrected_preview,
                    fit_ref,
                    min_variance=LINEAR_FIT_MIN_VARIANCE,
                )

                step += 1
                cycle_updates += 1
                _apply_phase2_correction(
                    gdf,
                    project_id=candidate,
                    slope=slope_applied,
                    intercept=intercept_applied,
                    step=step,
                    cycle=cycle,
                    reference_project=reference_project,
                )
                if CLIP_PHASE2_VALUES:
                    _clip_project_values(
                        gdf,
                        project_id=candidate,
                        min_value=PHASE2_MIN_VALUE,
                        max_value=clip_max_value,
                    )

                plot_rel = ""
                if SAVE_REGRESSION_PLOTS:
                    plot_name = (
                        f"cycle_{cycle:03d}__step_{step:05d}__ref_{safe_name(reference_project)}"
                        f"__cand_{safe_name(candidate)}.png"
                    )
                    plot_path = plot_dir / plot_name
                    save_regression_plot(
                        plot_path=plot_path,
                        fit_ref=fit_ref,
                        fit_cand=fit_cand,
                        slope=slope_applied,
                        intercept=intercept_applied,
                        post_slope=post_slope,
                        post_intercept=post_intercept,
                        reference_project=reference_project,
                        candidate_project=candidate,
                        full_slope=slope_full,
                        full_intercept=intercept_full,
                    )
                    plot_rel = str(plot_path.relative_to(OUTPUT_DIR))

                log_rows.append(
                    {
                        "cycle": cycle,
                        "step": step,
                        "status": "leveled_partial",
                        "reference_project": reference_project,
                        "candidate_project": candidate,
                        "reference_unique_count": ref_trimmed_unique_count,
                        "candidate_unique_count": cand_trimmed_unique_count,
                        "overlap_ref_points": int(len(ref_overlap_pos)),
                        "overlap_cand_points": int(len(cand_overlap_pos)),
                        "pair_count_raw": raw_count,
                        "pair_count_trimmed": fit_count,
                        "slope_full": slope_full,
                        "intercept_full": intercept_full,
                        "leveling_rate": PARTIAL_LEVELING_RATE,
                        "slope_applied": slope_applied,
                        "intercept_applied": intercept_applied,
                        "post_slope": post_slope,
                        "post_intercept": post_intercept,
                        "regression_plot": plot_rel,
                    }
                )

                print(
                    f"Cycle {cycle} step {step}: leveled {candidate} using {reference_project} "
                    f"(pairs={fit_count}/{raw_count}, full=({slope_full:.4f},{intercept_full:.4f}), "
                    f"applied=({slope_applied:.4f},{intercept_applied:.4f}))"
                )

        completed_cycles = cycle
        if cycle_updates < MIN_CYCLE_UPDATES:
            print(
                f"Stopping after cycle {cycle}: updates={cycle_updates} (< {MIN_CYCLE_UPDATES})."
            )
            break

    progress.close()

    if KEEP_ALL_PHASE2_INPUT_SURVEYS_IN_OUTPUT:
        final_out = gdf.copy()
    else:
        output_projects = set(levelable_projects).difference(excluded_low_unique_values)
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
    excluded_df = excluded_df[excluded_df["exclude_reason"] != ""].drop_duplicates(
        subset=[PROJECT_COLUMN], keep="last"
    )

    final_out_path = OUTPUT_DIR / PHASE2_OUTPUT_NAME
    final_out.to_file(final_out_path)

    log_df = pd.DataFrame(log_rows)
    log_path = OUTPUT_DIR / PHASE2_LOG_CSV_NAME
    log_df.to_csv(log_path, index=False)

    excluded_path = OUTPUT_DIR / PHASE2_EXCLUDED_CSV_NAME
    excluded_df.to_csv(excluded_path, index=False)

    leveled_count = (
        int((log_df["status"] == "leveled_partial").sum()) if not log_df.empty else 0
    )
    print(f"Saved phase-2 leveled shapefile: {final_out_path}")
    print(f"Saved phase-2 leveling log: {log_path}")
    print(f"Saved phase-2 excluded surveys: {excluded_path}")
    print(
        f"Phase-2 summary: total_surveys={len(survey_qa)}, levelable={len(levelable_projects)}, "
        f"excluded_quality={len(excluded_quality_projects)}, excluded_low_unique_values={len(excluded_low_unique_values)}, "
        f"leveled_partial_steps={leveled_count}, completed_cycles={completed_cycles}, "
        f"base_project={base_project}, keep_all_input={KEEP_ALL_PHASE2_INPUT_SURVEYS_IN_OUTPUT}"
    )


def main() -> None:
    print(f"Reading shapefile: {INPUT_SHP}")
    gdf = gpd.read_file(INPUT_SHP)
    print(f"Rows: {len(gdf)}")
    print("Columns:", ", ".join(map(str, gdf.columns)))

    validate_required_columns(
        gdf,
        {
            PROJECT_COLUMN,
            RAW_VALUE_COLUMN,
            IMPUTED_VALUE_COLUMN,
            CENSORED_COLUMN,
            LITHOLOGY_COLUMN,
        },
    )
    unknown_count = normalize_project_ids(gdf, PROJECT_COLUMN)
    if unknown_count:
        print(f"Normalized missing/None project IDs to 'Unknown': {unknown_count} rows")

    raw_nan_count, imp_nan_count = coerce_numeric_columns(
        gdf,
        RAW_VALUE_COLUMN,
        IMPUTED_VALUE_COLUMN,
        CENSORED_COLUMN,
    )
    print(
        f"Coerced numeric columns. NaN counts -> {RAW_VALUE_COLUMN}: {raw_nan_count}, "
        f"{IMPUTED_VALUE_COLUMN}: {imp_nan_count}"
    )

    survey_qa = build_survey_levelability(
        gdf,
        project_column=PROJECT_COLUMN,
        raw_value_column=RAW_VALUE_COLUMN,
        imputed_value_column=IMPUTED_VALUE_COLUMN,
        censored_column=CENSORED_COLUMN,
        min_pct_above_or_equal_lod=MIN_PCT_ABOVE_OR_EQUAL_LOD,
        max_pct_equal_lod=MAX_PCT_EQUAL_LOD,
        lod_equality_atol=LOD_EQUALITY_ATOL,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    qa_path = OUTPUT_DIR / PHASE2_QA_CSV_NAME
    survey_qa.to_csv(qa_path, index=False)
    print(f"Saved phase-2 survey levelability QA: {qa_path}")

    run_phase2_leveling(gdf, survey_qa)


if __name__ == "__main__":
    main()
