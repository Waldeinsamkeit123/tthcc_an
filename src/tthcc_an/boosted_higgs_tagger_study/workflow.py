from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tthcc_an.boosted_higgs_tagger_study.definitions import build_process_entries_from_summaries
from tthcc_an.boosted_higgs_tagger_study.inclusive_higgs_reporting import (
    plot_inclusive_higgs_wp,
    write_inclusive_higgs_wp_outputs,
)
from tthcc_an.boosted_higgs_tagger_study.y_split_reporting import (
    plot_y_split_study,
    write_y_split_outputs,
)
from tthcc_an.boosted_higgs_tagger_study.y_split_study import (
    build_y_split_aggregate,
    build_y_split_from_legacy_hcc,
)
from tthcc_an.boosted_higgs_tagger_study.inclusive_higgs_wp import (
    InclusiveHiggsCacheIncompatibleError,
    build_inclusive_higgs_wp_aggregate,
    build_inclusive_higgs_wp_from_existing_histogram,
    derive_inclusive_higgs_wp_results,
    historical_x_references,
)
from tthcc_an.boosted_higgs_tagger_study.hcc_wp_reporting import write_hcc_wp_outputs
from tthcc_an.boosted_higgs_tagger_study.plotting import (
    PlotStyle,
    plot_hcc_wp_scan_heatmaps,
)
from tthcc_an.boosted_higgs_tagger_study.strategy_reporting import (
    write_and_plot_strategy_comparison,
)
from tthcc_an.boosted_higgs_tagger_study.scans import (
    build_hcc_wp_scan_aggregate,
    HccScanCacheIncompatibleError,
    candidate_selection_metadata,
    current_hcc_wp_references,
    derive_event_level_results,
    derive_jet_level_results,
    validate_hcc_scan_provenance,
)


def write_hcc_wp_scan_study(
    aggregate: dict[str, Any],
    *,
    effective_args: Any,
    weighting_info: dict[str, Any],
    outdirs: dict[str, Path],
    plot_style: PlotStyle,
) -> dict[str, Any]:
    scan_config = effective_args.hcc_wp_scan
    if aggregate.get("config") != scan_config:
        raise HccScanCacheIncompatibleError(
            "Hcc WP scan grid/config in the cache does not match the requested JSON config."
        )
    validate_hcc_scan_provenance(weighting_info, effective_args)
    candidate_selection = candidate_selection_metadata(effective_args)
    current_references = current_hcc_wp_references()

    jet_result = derive_jet_level_results(aggregate)
    jet_summary = write_hcc_wp_outputs(
        jet_result,
        outdirs,
        scan_config,
        candidate_selection,
        current_references,
    )
    if not effective_args.skip_plots:
        plot_hcc_wp_scan_heatmaps(
            jet_result,
            outdirs["plots"] / "hcc_wp_scan",
            plot_style,
            current_references,
        )

    output = {
        "status": "available",
        "jet_level": jet_summary,
        "event_level": {
            "status": "disabled",
            "reason": "study.hcc_wp_scan.event_level_enabled is false",
        },
    }
    if scan_config["event_level_enabled"]:
        event_result = derive_event_level_results(aggregate)
        event_summary = write_hcc_wp_outputs(
            event_result,
            outdirs,
            scan_config,
            candidate_selection,
            current_references,
        )
        if not effective_args.skip_plots:
            plot_hcc_wp_scan_heatmaps(
                event_result,
                outdirs["plots"] / "hcc_wp_scan_event_level",
                plot_style,
                current_references,
            )
        output["event_level"] = event_summary

    return output

def maybe_write_hcc_wp_scan_from_raw(
    data: dict[str, Any],
    sample_summaries: list[dict[str, Any]],
    *,
    effective_args: Any,
    weighting_info: dict[str, Any],
    outdirs: dict[str, Path],
    plot_style: PlotStyle,
) -> dict[str, Any] | None:
    if not effective_args.hcc_wp_scan["enabled"]:
        return None
    aggregate = build_hcc_wp_scan_aggregate(
        data,
        build_process_entries_from_summaries(sample_summaries),
        effective_args.hcc_wp_scan,
    )
    return write_hcc_wp_scan_study(
        aggregate,
        effective_args=effective_args,
        weighting_info=weighting_info,
        outdirs=outdirs,
        plot_style=plot_style,
    )


def maybe_write_hcc_wp_scan_from_histogram(
    histogram_payload: dict[str, Any],
    *,
    effective_args: Any,
    weighting_info: dict[str, Any],
    outdirs: dict[str, Path],
    plot_style: PlotStyle,
) -> dict[str, Any] | None:
    if not effective_args.hcc_wp_scan["enabled"]:
        return None
    aggregate = histogram_payload.get("hcc_wp_scan")
    if aggregate is None:
        raise HccScanCacheIncompatibleError(
            "Hcc WP scan unavailable: this histogram cache predates the Hcc scan schema. "
            "Run a new ROOT scan or merge new chunk payloads."
        )
    return write_hcc_wp_scan_study(
        aggregate,
        effective_args=effective_args,
        weighting_info=weighting_info,
        outdirs=outdirs,
        plot_style=plot_style,
    )


def write_inclusive_higgs_wp_study(
    aggregate: dict[str, Any],
    *,
    effective_args: Any,
    weighting_info: dict[str, Any],
    outdirs: dict[str, Path],
    plot_style: PlotStyle,
) -> dict[str, Any]:
    config = effective_args.inclusive_higgs_wp
    if aggregate.get("config") != config:
        raise InclusiveHiggsCacheIncompatibleError(
            "Inclusive Higgs WP config in the cache does not match the requested JSON config."
        )
    validate_hcc_scan_provenance(weighting_info, effective_args)
    references = historical_x_references()
    result = derive_inclusive_higgs_wp_results(aggregate)
    summary = write_inclusive_higgs_wp_outputs(
        result,
        outdirs,
        candidate_selection_metadata(effective_args),
        references,
    )
    if not effective_args.skip_plots:
        plot_inclusive_higgs_wp(
            result,
            outdirs["plots"] / "inclusive_higgs_wp",
            plot_style,
            references,
        )
    return summary


def maybe_write_inclusive_higgs_wp_from_raw(
    data: dict[str, Any],
    sample_summaries: list[dict[str, Any]],
    *,
    effective_args: Any,
    weighting_info: dict[str, Any],
    outdirs: dict[str, Path],
    plot_style: PlotStyle,
) -> dict[str, Any] | None:
    if not effective_args.inclusive_higgs_wp["enabled"]:
        return None
    aggregate = build_inclusive_higgs_wp_aggregate(
        data,
        build_process_entries_from_summaries(sample_summaries),
        effective_args.inclusive_higgs_wp,
    )
    return write_inclusive_higgs_wp_study(
        aggregate,
        effective_args=effective_args,
        weighting_info=weighting_info,
        outdirs=outdirs,
        plot_style=plot_style,
    )


def maybe_write_inclusive_higgs_wp_from_histogram(
    histogram_payload: dict[str, Any],
    *,
    effective_args: Any,
    weighting_info: dict[str, Any],
    outdirs: dict[str, Path],
    plot_style: PlotStyle,
) -> dict[str, Any] | None:
    if not effective_args.inclusive_higgs_wp["enabled"]:
        return None
    aggregate = histogram_payload.get("inclusive_higgs_wp")
    if aggregate is None:
        aggregate = build_inclusive_higgs_wp_from_existing_histogram(
            histogram_payload,
            effective_args.inclusive_higgs_wp,
        )
    return write_inclusive_higgs_wp_study(
        aggregate,
        effective_args=effective_args,
        weighting_info=weighting_info,
        outdirs=outdirs,
        plot_style=plot_style,
    )



def _validate_y_split_x_consistency(
    inclusive_aggregate: dict[str, Any],
    y_split_aggregate: dict[str, Any],
) -> None:
    inclusive_result = derive_inclusive_higgs_wp_results(inclusive_aggregate)
    recommendations = {
        float(row["target_non_higgs_jet_efficiency"]): float(row["x_cut"])
        for row in inclusive_result["recommendations"]
    }
    for working_point in y_split_aggregate["working_points"]:
        target = float(working_point["non_higgs_target_efficiency"])
        matches = [
            x_cut
            for configured_target, x_cut in recommendations.items()
            if abs(configured_target - target) <= 1.0e-12
        ]
        if len(matches) != 1 or not np.isclose(
            matches[0], float(working_point["x_cut"]), rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                "Unified boosted-tagger payload has inconsistent Inclusive/y-split "
                f"x WP for non-Higgs target {target:g}: inclusive={matches}, "
                f"y_split={working_point['x_cut']}."
            )

def write_y_split_study(
    aggregate: dict[str, Any],
    *,
    effective_args: Any,
    weighting_info: dict[str, Any],
    outdirs: dict[str, Path],
    plot_style: PlotStyle,
) -> dict[str, Any]:
    if aggregate.get("config") != effective_args.y_split_study:
        raise ValueError("y-split cache config does not match the requested config.")
    validate_hcc_scan_provenance(weighting_info, effective_args)
    summary = write_y_split_outputs(
        aggregate,
        outdirs,
        candidate_selection_metadata(effective_args),
    )
    if not effective_args.skip_plots:
        plot_y_split_study(
            aggregate,
            summary["scan"],
            outdirs["plots"] / "y_split_study",
            plot_style,
        )
    return summary


def maybe_write_y_split_from_raw(
    data: dict[str, Any],
    sample_summaries: list[dict[str, Any]],
    *,
    effective_args: Any,
    weighting_info: dict[str, Any],
    outdirs: dict[str, Path],
    plot_style: PlotStyle,
) -> dict[str, Any] | None:
    if not effective_args.y_split_study["enabled"]:
        return None
    aggregate = build_y_split_aggregate(
        data,
        build_process_entries_from_summaries(sample_summaries),
        effective_args.y_split_study,
    )
    return write_y_split_study(
        aggregate,
        effective_args=effective_args,
        weighting_info=weighting_info,
        outdirs=outdirs,
        plot_style=plot_style,
    )


def maybe_write_y_split_from_histogram(
    histogram_payload: dict[str, Any],
    *,
    effective_args: Any,
    weighting_info: dict[str, Any],
    outdirs: dict[str, Path],
    plot_style: PlotStyle,
) -> dict[str, Any] | None:
    if not effective_args.y_split_study["enabled"]:
        return None
    aggregate = histogram_payload.get("y_split_study")
    has_dedicated_aggregate = aggregate is not None
    if has_dedicated_aggregate and histogram_payload.get("inclusive_higgs_wp") is not None:
        _validate_y_split_x_consistency(
            histogram_payload["inclusive_higgs_wp"], aggregate
        )
    if aggregate is None:
        hcc = histogram_payload.get("hcc_wp_scan")
        if hcc is None or not hcc.get("event_level_available", False):
            raise ValueError(
                "Legacy payload lacks the Hcc scan needed for y-split diagnostics."
            )
        aggregate = build_y_split_from_legacy_hcc(
            hcc, effective_args.y_split_study
        )
    summary = write_y_split_study(
        aggregate,
        effective_args=effective_args,
        weighting_info=weighting_info,
        outdirs=outdirs,
        plot_style=plot_style,
    )
    if has_dedicated_aggregate:
        summary["strategy_comparison"] = write_and_plot_strategy_comparison(
            histogram_payload,
            outdirs,
            plot_style,
            skip_plots=effective_args.skip_plots,
        )
    return summary
