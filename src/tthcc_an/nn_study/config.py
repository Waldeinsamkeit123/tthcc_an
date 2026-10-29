from __future__ import annotations

import ast
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from tthcc_an.config_loader import expand_file_patterns, load_json_maybe_with_comments


DEFAULT_TREE_NAME = "Events"
DEFAULT_WEIGHT_BRANCH = "weight"
DEFAULT_UPROOT_STEP_SIZE = "200 MB"
DEFAULT_OUTDIR = "outputs/nn_study"


@dataclass(frozen=True)
class NnScoreClass:
    name: str
    label: str
    branch: str
    color: str | None


@dataclass(frozen=True)
class NnTruthCategory:
    name: str
    label: str
    expression: str
    color: str | None


@dataclass(frozen=True)
class NnSample:
    name: str
    dataset: str
    label: str
    files: list[str]
    selection: str


@dataclass(frozen=True)
class NnMassVariable:
    branch: str
    label: str
    bins: int
    range_quantiles: tuple[float, float]
    padding_fraction: float


@dataclass(frozen=True)
class NnMassScan:
    score_name: str
    score_branch: str
    direction: str
    thresholds: list[float]
    populations: list[str] | None = None


@dataclass(frozen=True)
class NnMassPopulation:
    name: str
    label: str
    truth_categories: list[str] | None
    samples: list[str] | None
    exclude_samples: list[str]
    sample_labels: list[str] | None
    exclude_sample_labels: list[str]


@dataclass(frozen=True)
class NnMassSculptingConfig:
    enabled: bool
    populations: list[NnMassPopulation]
    variables: list[NnMassVariable]
    scans: list[NnMassScan]
    summary_scans: list[NnMassScan]


@dataclass(frozen=True)
class NnScoreSignificanceScan:
    score_name: str
    score_branch: str
    direction: str
    scan_min: float
    scan_max: float
    scan_points: int


@dataclass(frozen=True)
class NnScoreSignificanceConfig:
    enabled: bool
    signals: list[str]
    metric: str
    scans: list[NnScoreSignificanceScan]


@dataclass(frozen=True)
class NnScoreSignificance2DAxis:
    score_name: str
    score_branch: str
    direction: str
    thresholds: list[float]


@dataclass(frozen=True)
class NnScoreSignificance2DGroup:
    name: str
    label: str
    truth_categories: list[str]


@dataclass(frozen=True)
class NnScoreSignificance2DThirdCut:
    enabled: bool
    score_name: str | None
    score_branch: str | None
    direction: str | None
    thresholds: list[float]
    base_points: list[tuple[float, float]]


@dataclass(frozen=True)
class NnScoreSignificance2DConfig:
    enabled: bool
    signals: list[str]
    metric: str
    x_axis: NnScoreSignificance2DAxis | None
    y_axis: NnScoreSignificance2DAxis | None
    groups: list[NnScoreSignificance2DGroup]
    representative_points: list[tuple[float, float]]
    plateau_fractions: list[float]
    third_cut: NnScoreSignificance2DThirdCut


@dataclass(frozen=True)
class NnSignificanceMassWindowConfig:
    enabled: bool
    branch: str | None
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True)
class NnAnalysisRegionCut:
    score_name: str
    score_branch: str
    direction: str
    threshold: float


@dataclass(frozen=True)
class NnConditionalQcdVetoConfig:
    enabled: bool
    tthz_score_name: str | None
    tthz_score_branch: str | None
    qcd_score_name: str | None
    qcd_score_branch: str | None
    tthz_thresholds: list[float]
    qcd_thresholds: list[float]


@dataclass(frozen=True)
class NnArScorePlotRange:
    score_name: str
    score_branch: str
    minimum: float
    maximum: float
    bins: int


@dataclass(frozen=True)
class NnAnalysisRegionsConfig:
    enabled: bool
    ar_cuts: list[NnAnalysisRegionCut]
    ar_score_plot_ranges: dict[str, NnArScorePlotRange]
    conditional_qcd_veto: NnConditionalQcdVetoConfig


@dataclass(frozen=True)
class NnQcdScoreGroup:
    name: str
    label: str
    truth_categories: list[str]
    color: str | None


@dataclass(frozen=True)
class NnQcdScoreScanConfig:
    enabled: bool
    score_name: str
    score_branch: str
    direction: str
    candidate_thresholds: list[float]
    scan_min: float
    scan_max: float
    scan_points: int
    scan_scale: str
    distribution_bins: int
    distribution_log10_range: tuple[float, float]
    groups: list[NnQcdScoreGroup]
    working_point_bins: int
    working_point_score_range: tuple[float, float]
    reference_threshold: float | None
    working_point_scale: str
    distribution_groups: list[NnQcdScoreGroup]
    significance_enabled: bool
    significance_signals: list[str]
    significance_metric: str


@dataclass(frozen=True)
class NnWeightingDiagnosticsConfig:
    enabled: bool
    reweight_variables: dict[str, list[float]]
    class_weights: dict[str, float]
    key_auc_pairs: list[tuple[str, str]]
    model_output_classes: list[str]
    persisted_score_classes: list[str]


@dataclass(frozen=True)
class NnStudyConfig:
    config_path: Path
    repo_root: Path
    channel: str
    input_location: str
    sample_file_pattern: str | None
    outdir: Path
    tree_name: str
    weight_branch: str
    selection: str
    selection_branches: list[str]
    truth_branches: list[str]
    analysis_branches: list[str]
    flatten_first_branches: list[str]
    scores: list[NnScoreClass]
    auxiliary_scores: list[NnScoreClass]
    analysis_discriminants: list[str]
    truth_categories: list[NnTruthCategory]
    prediction_truth_categories: list[NnTruthCategory]
    diagnostic_truth_categories: list[NnTruthCategory]
    samples: list[NnSample]
    gen_sumw_file: str
    xsec_file: str
    lumi_fb: float
    uproot_step_size: str
    max_files_per_sample: int | None
    plot_options: dict[str, Any]
    regressed_mass_fine_binning: dict[str, Any]
    qcd_data_mc: dict[str, Any]
    tthz_qcd_discriminant_enabled: bool
    mass_sculpting: NnMassSculptingConfig
    score_significance: NnScoreSignificanceConfig
    score_significance_2d: NnScoreSignificance2DConfig
    score_significance_2d_hierarchical: NnScoreSignificance2DConfig
    significance_mass_window: NnSignificanceMassWindowConfig
    analysis_regions: NnAnalysisRegionsConfig
    qcd_score_scan: NnQcdScoreScanConfig
    weighting_diagnostics: NnWeightingDiagnosticsConfig
    validate_auc_with_sklearn: bool
    year: str
    energy_tev: float

    @property
    def score_names(self) -> list[str]:
        return [score.name for score in self.scores]

    @property
    def all_scores(self) -> list[NnScoreClass]:
        return [*self.scores, *self.auxiliary_scores]

    @property
    def all_score_names(self) -> list[str]:
        return [score.name for score in self.all_scores]

    @property
    def score_branches(self) -> list[str]:
        return [score.branch for score in self.all_scores]

    @property
    def truth_names(self) -> list[str]:
        return [truth.name for truth in self.truth_categories]

    @property
    def prediction_truth_names(self) -> list[str]:
        return [truth.name for truth in self.prediction_truth_categories]

    @property
    def uses_distinct_prediction_truth(self) -> bool:
        return self.prediction_truth_categories != self.truth_categories

    @property
    def requested_branches(self) -> list[str]:
        return self.requested_branches_for(
            score_names=self.all_score_names,
            analysis_branches=self.analysis_branches,
        )

    def requested_branches_for(
        self,
        *,
        score_names: list[str],
        analysis_branches: list[str],
    ) -> list[str]:
        score_by_name = {score.name: score for score in self.all_scores}
        unknown_scores = sorted(set(score_names) - set(score_by_name))
        if unknown_scores:
            raise KeyError(f"Unknown requested score names: {', '.join(unknown_scores)}")
        branches = {score_by_name[name].branch for name in score_names}
        branches.update(self.truth_branches)
        branches.update(self.selection_branches)
        branches.update(analysis_branches)
        branches.update(_expression_branch_names(self.selection))
        for sample in self.samples:
            branches.update(_expression_branch_names(sample.selection))
        for truth in self.truth_categories:
            branches.update(_expression_branch_names(truth.expression))
        for truth in self.prediction_truth_categories:
            branches.update(_expression_branch_names(truth.expression))
        for truth in self.diagnostic_truth_categories:
            branches.update(_expression_branch_names(truth.expression))
        branches.add(self.weight_branch)
        return sorted(branches)


def _resolve_path(path: str | Path, anchor: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (anchor / candidate).resolve()


def _expression_branch_names(expression: str) -> set[str]:
    if not expression.strip():
        return set()
    tree = ast.parse(expression, mode="eval")
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id != "np"
    }


def _resolve_outdir(repo_root: Path, configured: str | None, explicit: str | None) -> Path:
    candidate = Path(explicit or configured or DEFAULT_OUTDIR)
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def _unique_names(values: list[str], kind: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"Duplicate {kind} names: {', '.join(duplicates)}")


def _load_mass_sculpting(
    payload: dict[str, Any],
    scores: list[NnScoreClass],
    truths: list[NnTruthCategory],
    samples: list[NnSample],
) -> NnMassSculptingConfig:
    mass_payload = dict(payload.get("mass_sculpting", {}))
    raw_populations = mass_payload.get("populations")
    if raw_populations is None:
        raw_populations = [mass_payload.get("population", {})]
    populations: list[NnMassPopulation] = []
    for raw_population in list(raw_populations):
        population_payload = dict(raw_population)
        raw_truth_categories = population_payload.get("truth_categories", "all")
        if raw_truth_categories == "all":
            population_truths = None
        else:
            population_truths = [str(name) for name in list(raw_truth_categories)]
            unknown_truths = sorted(
                set(population_truths) - {truth.name for truth in truths}
            )
            if unknown_truths:
                raise ValueError(
                    "Unknown mass-sculpting truth categories: "
                    + ", ".join(unknown_truths)
                )
        raw_samples = population_payload.get("samples", "all")
        if raw_samples == "all":
            population_samples = None
        else:
            population_samples = [str(name) for name in list(raw_samples)]
            unknown_samples = sorted(
                set(population_samples) - {sample.name for sample in samples}
            )
            if unknown_samples:
                raise ValueError(
                    "Unknown mass-sculpting samples: " + ", ".join(unknown_samples)
                )
        excluded_samples = [
            str(name) for name in list(population_payload.get("exclude_samples", []))
        ]
        unknown_excluded_samples = sorted(
            set(excluded_samples) - {sample.name for sample in samples}
        )
        if unknown_excluded_samples:
            raise ValueError(
                "Unknown excluded mass-sculpting samples: "
                + ", ".join(unknown_excluded_samples)
            )
        if len(excluded_samples) != len(set(excluded_samples)):
            raise ValueError("Excluded mass-sculpting samples must be unique.")
        available_sample_labels = {sample.label for sample in samples}
        raw_sample_labels = population_payload.get("sample_labels", "all")
        if raw_sample_labels == "all":
            population_sample_labels = None
        else:
            population_sample_labels = [
                str(label) for label in list(raw_sample_labels)
            ]
            unknown_sample_labels = sorted(
                set(population_sample_labels) - available_sample_labels
            )
            if unknown_sample_labels:
                raise ValueError(
                    "Unknown mass-sculpting sample labels: "
                    + ", ".join(unknown_sample_labels)
                )
        excluded_sample_labels = [
            str(label)
            for label in list(population_payload.get("exclude_sample_labels", []))
        ]
        unknown_excluded_sample_labels = sorted(
            set(excluded_sample_labels) - available_sample_labels
        )
        if unknown_excluded_sample_labels:
            raise ValueError(
                "Unknown excluded mass-sculpting sample labels: "
                + ", ".join(unknown_excluded_sample_labels)
            )
        if len(excluded_sample_labels) != len(set(excluded_sample_labels)):
            raise ValueError("Excluded mass-sculpting sample labels must be unique.")
        populations.append(
            NnMassPopulation(
                name=str(population_payload.get("name", "all_selected_mc")),
                label=str(population_payload.get("label", "All selected MC")),
                truth_categories=population_truths,
                samples=population_samples,
                exclude_samples=excluded_samples,
                sample_labels=population_sample_labels,
                exclude_sample_labels=excluded_sample_labels,
            )
        )
    _unique_names([population.name for population in populations], "mass population")

    variables: list[NnMassVariable] = []
    for entry in list(mass_payload.get("variables", [])):
        quantiles = [float(value) for value in entry.get("range_quantiles", [0.005, 0.995])]
        if len(quantiles) != 2 or not (0.0 <= quantiles[0] < quantiles[1] <= 1.0):
            raise ValueError("Mass-sculpting range_quantiles must contain two ordered values in [0, 1].")
        bins = int(entry.get("bins", 45))
        padding_fraction = float(entry.get("padding_fraction", 0.04))
        if bins <= 0:
            raise ValueError("Mass-sculpting bins must be positive.")
        if padding_fraction < 0.0:
            raise ValueError("Mass-sculpting padding_fraction must be non-negative.")
        branch = str(entry["branch"])
        variables.append(
            NnMassVariable(
                branch=branch,
                label=str(entry.get("label", branch)),
                bins=bins,
                range_quantiles=(quantiles[0], quantiles[1]),
                padding_fraction=padding_fraction,
            )
        )

    score_by_reference: dict[str, NnScoreClass] = {}
    for score in scores:
        score_by_reference[score.name] = score
        score_by_reference[score.branch] = score
    population_names = {population.name for population in populations}

    def load_scans(key: str) -> list[NnMassScan]:
        scans: list[NnMassScan] = []
        for entry in list(mass_payload.get(key, [])):
            reference = str(entry["score"])
            if reference not in score_by_reference:
                raise ValueError(f"Unknown mass-sculpting score '{reference}'.")
            score = score_by_reference[reference]
            direction = str(entry["direction"])
            if direction not in {">", ">=", "<", "<="}:
                raise ValueError(
                    "Mass-sculpting direction must be one of >, >=, <, <=."
                )
            thresholds = [float(value) for value in list(entry.get("thresholds", []))]
            if not thresholds:
                raise ValueError(f"Mass-sculpting scan '{reference}' has no thresholds.")
            if thresholds != sorted(thresholds):
                raise ValueError(
                    f"Mass-sculpting thresholds for '{reference}' must be ascending."
                )
            raw_scan_populations = entry.get("populations", "all")
            scan_populations = (
                None
                if raw_scan_populations == "all"
                else [str(name) for name in list(raw_scan_populations)]
            )
            unknown_populations = sorted(
                set(scan_populations or []) - population_names
            )
            if unknown_populations:
                raise ValueError(
                    "Unknown mass-sculpting scan populations: "
                    + ", ".join(unknown_populations)
                )
            scans.append(
                NnMassScan(
                    score_name=score.name,
                    score_branch=score.branch,
                    direction=direction,
                    thresholds=thresholds,
                    populations=scan_populations,
                )
            )
        return scans

    scans = load_scans("scans")
    summary_scans = load_scans("summary_scans")

    enabled = bool(mass_payload.get("enabled", False))
    if enabled and (not populations or not variables):
        raise ValueError(
            "Enabled mass_sculpting requires non-empty populations and variables."
        )
    return NnMassSculptingConfig(
        enabled=enabled,
        populations=populations,
        variables=variables,
        scans=scans,
        summary_scans=summary_scans,
    )


def _load_score_significance(
    payload: dict[str, Any],
    scores: list[NnScoreClass],
    truths: list[NnTruthCategory],
) -> NnScoreSignificanceConfig:
    significance_payload = dict(payload.get("score_significance", {}))
    enabled = bool(significance_payload.get("enabled", False))
    metric = str(
        significance_payload.get("metric", "s_over_sqrt_s_plus_b")
    )
    if metric != "s_over_sqrt_s_plus_b":
        raise ValueError(
            "score_significance.metric must be 's_over_sqrt_s_plus_b'."
        )

    truth_names = {truth.name for truth in truths}
    signals = [
        str(name) for name in list(significance_payload.get("signals", []))
    ]
    unknown_signals = sorted(set(signals) - truth_names)
    if unknown_signals:
        raise ValueError(
            "Unknown score-significance signals: " + ", ".join(unknown_signals)
        )
    if len(signals) != len(set(signals)):
        raise ValueError("Score-significance signals must be unique.")

    score_by_reference: dict[str, NnScoreClass] = {}
    for score in scores:
        score_by_reference[score.name] = score
        score_by_reference[score.branch] = score
    scans: list[NnScoreSignificanceScan] = []
    for entry in list(significance_payload.get("scans", [])):
        reference = str(entry["score"])
        if reference not in score_by_reference:
            raise ValueError(f"Unknown score-significance score '{reference}'.")
        score = score_by_reference[reference]
        direction = str(entry["direction"])
        if direction not in {"<", ">"}:
            raise ValueError("Score-significance direction must be '<' or '>'.")
        scan_min = float(entry["min"])
        scan_max = float(entry["max"])
        scan_points = int(entry["points"])
        if not scan_min < scan_max:
            raise ValueError(
                f"Score-significance scan '{reference}' requires min < max."
            )
        if scan_points < 2:
            raise ValueError(
                f"Score-significance scan '{reference}' needs at least two points."
            )
        scans.append(
            NnScoreSignificanceScan(
                score_name=score.name,
                score_branch=score.branch,
                direction=direction,
                scan_min=scan_min,
                scan_max=scan_max,
                scan_points=scan_points,
            )
        )
    _unique_names([scan.score_name for scan in scans], "score-significance scan")
    if enabled and (not signals or not scans):
        raise ValueError(
            "Enabled score_significance requires non-empty signals and scans."
        )
    return NnScoreSignificanceConfig(
        enabled=enabled,
        signals=signals,
        metric=metric,
        scans=scans,
    )


def _load_score_significance_2d(
    payload: dict[str, Any],
    scores: list[NnScoreClass],
    truths: list[NnTruthCategory],
    *,
    config_key: str = "score_significance_2d",
) -> NnScoreSignificance2DConfig:
    scan_payload = dict(payload.get(config_key, {}))
    enabled = bool(scan_payload.get("enabled", False))
    metric = str(scan_payload.get("metric", "s_over_sqrt_s_plus_b"))
    if metric != "s_over_sqrt_s_plus_b":
        raise ValueError(
            "score_significance_2d.metric must be 's_over_sqrt_s_plus_b'."
        )

    score_by_reference: dict[str, NnScoreClass] = {}
    for score in scores:
        score_by_reference[score.name] = score
        score_by_reference[score.branch] = score

    def load_axis(key: str) -> NnScoreSignificance2DAxis | None:
        raw_axis = scan_payload.get(key)
        if raw_axis is None:
            return None
        axis_payload = dict(raw_axis)
        reference = str(axis_payload["score"])
        if reference not in score_by_reference:
            raise ValueError(
                f"Unknown score-significance-2d {key} score '{reference}'."
            )
        score = score_by_reference[reference]
        direction = str(axis_payload["direction"])
        if direction not in {"<", ">"}:
            raise ValueError(
                f"score_significance_2d.{key}.direction must be '<' or '>'."
            )
        thresholds = [
            float(value) for value in list(axis_payload.get("thresholds", []))
        ]
        if (
            not thresholds
            or thresholds != sorted(thresholds)
            or len(thresholds) != len(set(thresholds))
            or not all(isfinite(value) for value in thresholds)
        ):
            raise ValueError(
                f"score_significance_2d.{key}.thresholds must be non-empty, "
                "finite, unique, and ascending."
            )
        return NnScoreSignificance2DAxis(
            score_name=score.name,
            score_branch=score.branch,
            direction=direction,
            thresholds=thresholds,
        )

    x_axis = load_axis("x")
    y_axis = load_axis("y")
    truth_names = {truth.name for truth in truths}
    signals = [str(name) for name in list(scan_payload.get("signals", []))]
    unknown_signals = sorted(set(signals) - truth_names)
    if unknown_signals:
        raise ValueError(
            "Unknown score-significance-2d signals: " + ", ".join(unknown_signals)
        )
    if len(signals) != len(set(signals)):
        raise ValueError("score_significance_2d signals must be unique.")

    groups: list[NnScoreSignificance2DGroup] = []
    for entry in list(scan_payload.get("groups", [])):
        categories = [str(name) for name in list(entry.get("truth_categories", []))]
        unknown_categories = sorted(set(categories) - truth_names)
        if unknown_categories:
            raise ValueError(
                "Unknown score-significance-2d group truth categories: "
                + ", ".join(unknown_categories)
            )
        if not categories:
            raise ValueError(
                "Each score_significance_2d group requires truth_categories."
            )
        groups.append(
            NnScoreSignificance2DGroup(
                name=str(entry["name"]),
                label=str(entry.get("label", entry["name"])),
                truth_categories=categories,
            )
        )
    _unique_names([group.name for group in groups], "score-significance-2d group")

    def load_points(raw_points: list[Any], key: str) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for raw_point in raw_points:
            if isinstance(raw_point, dict):
                point = (float(raw_point["x"]), float(raw_point["y"]))
            else:
                values = [float(value) for value in list(raw_point)]
                if len(values) != 2:
                    raise ValueError(
                        f"Each score_significance_2d.{key} entry needs x and y."
                    )
                point = (values[0], values[1])
            if point in points:
                raise ValueError(
                    f"score_significance_2d.{key} points must be unique."
                )
            points.append(point)
        return points

    representative_points = load_points(
        list(scan_payload.get("representative_points", [])),
        "representative_points",
    )
    plateau_fractions = [
        float(value)
        for value in list(scan_payload.get("plateau_fractions", [0.95, 0.98]))
    ]
    if (
        plateau_fractions != sorted(plateau_fractions)
        or len(plateau_fractions) != len(set(plateau_fractions))
        or any(not 0.0 < value <= 1.0 for value in plateau_fractions)
    ):
        raise ValueError(
            "score_significance_2d.plateau_fractions must be unique, ascending, "
            "and in (0, 1]."
        )

    third_payload = dict(scan_payload.get("third_score_diagnostic", {}))
    third_enabled = bool(third_payload.get("enabled", False))
    third_reference = third_payload.get("score")
    if third_reference is None:
        third_score = None
    elif str(third_reference) not in score_by_reference:
        raise ValueError(
            f"Unknown third-score diagnostic score '{third_reference}'."
        )
    else:
        third_score = score_by_reference[str(third_reference)]
    third_direction = (
        None
        if third_payload.get("direction") is None
        else str(third_payload["direction"])
    )
    if third_direction is not None and third_direction not in {"<", ">"}:
        raise ValueError(
            "score_significance_2d.third_score_diagnostic.direction must be '<' or '>'."
        )
    third_thresholds = [
        float(value) for value in list(third_payload.get("thresholds", []))
    ]
    if (
        third_thresholds != sorted(third_thresholds)
        or len(third_thresholds) != len(set(third_thresholds))
        or not all(isfinite(value) for value in third_thresholds)
    ):
        raise ValueError(
            "Third-score thresholds must be finite, unique, and ascending."
        )
    third_base_points = load_points(
        list(third_payload.get("base_points", representative_points)),
        "third_score_diagnostic.base_points",
    )
    third_cut = NnScoreSignificance2DThirdCut(
        enabled=third_enabled,
        score_name=None if third_score is None else third_score.name,
        score_branch=None if third_score is None else third_score.branch,
        direction=third_direction,
        thresholds=third_thresholds,
        base_points=third_base_points,
    )

    if enabled:
        if x_axis is None or y_axis is None or not signals or not groups:
            raise ValueError(
                "Enabled score_significance_2d requires x/y axes, signals, and groups."
            )
        missing_signal_groups = sorted(
            set(signals) - {group.name for group in groups}
        )
        if missing_signal_groups:
            raise ValueError(
                "score_significance_2d requires a matching efficiency group for "
                "each signal: " + ", ".join(missing_signal_groups)
            )
        for key, points in (
            ("representative_points", representative_points),
            ("third_score_diagnostic.base_points", third_base_points),
        ):
            for x_value, y_value in points:
                if x_value not in x_axis.thresholds or y_value not in y_axis.thresholds:
                    raise ValueError(
                        f"score_significance_2d.{key} point "
                        f"({x_value:g}, {y_value:g}) is not on the configured grid."
                    )
        if third_enabled and (
            third_score is None
            or third_direction is None
            or not third_thresholds
            or not third_base_points
        ):
            raise ValueError(
                "Enabled third-score diagnostic requires score, direction, "
                "thresholds, and base_points."
            )

    return NnScoreSignificance2DConfig(
        enabled=enabled,
        signals=signals,
        metric=metric,
        x_axis=x_axis,
        y_axis=y_axis,
        groups=groups,
        representative_points=representative_points,
        plateau_fractions=plateau_fractions,
        third_cut=third_cut,
    )


def _load_significance_mass_window(
    payload: dict[str, Any],
) -> NnSignificanceMassWindowConfig:
    window_payload = dict(payload.get("significance_mass_window", {}))
    enabled = bool(window_payload.get("enabled", False))
    branch_value = window_payload.get("branch")
    branch = None if branch_value is None else str(branch_value)
    raw_range = [float(value) for value in list(window_payload.get("range", []))]
    if enabled and not branch:
        raise ValueError(
            "Enabled significance_mass_window requires a non-empty branch."
        )
    if enabled and (
        len(raw_range) != 2
        or not all(isfinite(value) for value in raw_range)
        or raw_range[0] >= raw_range[1]
    ):
        raise ValueError(
            "Enabled significance_mass_window.range must contain two ascending values."
        )
    if not enabled:
        return NnSignificanceMassWindowConfig(
            enabled=False,
            branch=None,
            minimum=None,
            maximum=None,
        )
    return NnSignificanceMassWindowConfig(
        enabled=True,
        branch=branch,
        minimum=raw_range[0],
        maximum=raw_range[1],
    )


def _load_analysis_regions(
    payload: dict[str, Any], scores: list[NnScoreClass]
) -> NnAnalysisRegionsConfig:
    region_payload = dict(payload.get("analysis_regions", {}))
    enabled = bool(region_payload.get("enabled", False))
    score_by_reference: dict[str, NnScoreClass] = {}
    for score in scores:
        score_by_reference[score.name] = score
        score_by_reference[score.branch] = score

    ar_payload = dict(region_payload.get("ar", {}))
    ar_cuts: list[NnAnalysisRegionCut] = []
    for entry in list(ar_payload.get("cuts", [])):
        reference = str(entry["score"])
        if reference not in score_by_reference:
            raise ValueError(f"Unknown analysis-region score '{reference}'.")
        direction = str(entry["direction"])
        if direction not in {"<", "<=", ">", ">="}:
            raise ValueError(
                "Analysis-region direction must be one of <, <=, >, >=."
            )
        threshold = float(entry["threshold"])
        if not isfinite(threshold):
            raise ValueError("Analysis-region thresholds must be finite.")
        score = score_by_reference[reference]
        ar_cuts.append(
            NnAnalysisRegionCut(
                score_name=score.name,
                score_branch=score.branch,
                direction=direction,
                threshold=threshold,
            )
        )
    _unique_names([cut.score_name for cut in ar_cuts], "analysis-region cut score")
    if enabled and not ar_cuts:
        raise ValueError("Enabled analysis_regions requires non-empty ar.cuts.")

    ar_score_plot_ranges: dict[str, NnArScorePlotRange] = {}
    for reference, entry in dict(ar_payload.get("score_plot_ranges", {})).items():
        if reference not in score_by_reference:
            raise ValueError(f"Unknown AR score plot range '{reference}'.")
        minimum = float(entry["min"])
        maximum = float(entry["max"])
        bins = int(entry["bins"])
        if not (isfinite(minimum) and isfinite(maximum) and minimum < maximum):
            raise ValueError(
                f"AR score plot range '{reference}' requires finite min < max."
            )
        if bins <= 0:
            raise ValueError(f"AR score plot range '{reference}' requires bins > 0.")
        score = score_by_reference[reference]
        if score.name in ar_score_plot_ranges:
            raise ValueError(f"Duplicate AR score plot range for '{score.name}'.")
        ar_score_plot_ranges[score.name] = NnArScorePlotRange(
            score_name=score.name,
            score_branch=score.branch,
            minimum=minimum,
            maximum=maximum,
            bins=bins,
        )

    conditional_payload = dict(region_payload.get("conditional_qcd_veto", {}))
    conditional_enabled = bool(conditional_payload.get("enabled", False))

    def resolve_conditional_score(key: str) -> NnScoreClass | None:
        reference = conditional_payload.get(key)
        if reference is None:
            return None
        reference = str(reference)
        if reference not in score_by_reference:
            raise ValueError(
                f"Unknown conditional-QCD score '{reference}' for '{key}'."
            )
        return score_by_reference[reference]

    tthz_score = resolve_conditional_score("tthz_score")
    qcd_score = resolve_conditional_score("qcd_score")
    tthz_thresholds = [
        float(value)
        for value in list(conditional_payload.get("tthz_thresholds", []))
    ]
    qcd_thresholds = [
        float(value)
        for value in list(conditional_payload.get("qcd_thresholds", []))
    ]
    for name, values in (
        ("tthz_thresholds", tthz_thresholds),
        ("qcd_thresholds", qcd_thresholds),
    ):
        if (
            values != sorted(values)
            or len(values) != len(set(values))
            or not all(isfinite(value) for value in values)
        ):
            raise ValueError(
                f"analysis_regions.conditional_qcd_veto.{name} must be finite, "
                "unique, and ascending."
            )
    if conditional_enabled and (
        tthz_score is None
        or qcd_score is None
        or not tthz_thresholds
        or not qcd_thresholds
    ):
        raise ValueError(
            "Enabled conditional_qcd_veto requires tthz/qcd scores and thresholds."
        )
    return NnAnalysisRegionsConfig(
        enabled=enabled,
        ar_cuts=ar_cuts,
        ar_score_plot_ranges=ar_score_plot_ranges,
        conditional_qcd_veto=NnConditionalQcdVetoConfig(
            enabled=conditional_enabled,
            tthz_score_name=None if tthz_score is None else tthz_score.name,
            tthz_score_branch=None if tthz_score is None else tthz_score.branch,
            qcd_score_name=None if qcd_score is None else qcd_score.name,
            qcd_score_branch=None if qcd_score is None else qcd_score.branch,
            tthz_thresholds=tthz_thresholds,
            qcd_thresholds=qcd_thresholds,
        ),
    )


def _load_qcd_score_scan(
    payload: dict[str, Any],
    scores: list[NnScoreClass],
    truths: list[NnTruthCategory],
) -> NnQcdScoreScanConfig:
    scan_payload = dict(payload.get("qcd_score_scan", {}))
    enabled = bool(scan_payload.get("enabled", False))
    score_by_reference: dict[str, NnScoreClass] = {}
    for score in scores:
        score_by_reference[score.name] = score
        score_by_reference[score.branch] = score
    reference = str(scan_payload.get("score", "qcd"))
    if reference not in score_by_reference:
        if enabled:
            raise ValueError(f"Unknown QCD-score scan score '{reference}'.")
        score = scores[0]
    else:
        score = score_by_reference[reference]

    direction = str(scan_payload.get("direction", "<"))
    if direction != "<":
        raise ValueError("QCD-score scan direction must be '<'.")
    candidate_thresholds = [
        float(value) for value in list(scan_payload.get("candidate_thresholds", []))
    ]
    if candidate_thresholds != sorted(candidate_thresholds):
        raise ValueError("QCD-score candidate thresholds must be ascending.")
    if len(candidate_thresholds) != len(set(candidate_thresholds)):
        raise ValueError("QCD-score candidate thresholds must be unique.")
    scan_min = float(scan_payload.get("scan_min", 1e-6))
    scan_max = float(scan_payload.get("scan_max", 1.0))
    scan_points = int(scan_payload.get("scan_points", 100))
    scan_scale = str(scan_payload.get("scan_scale", "log"))
    if scan_scale not in {"linear", "log"}:
        raise ValueError("QCD-score scan_scale must be 'linear' or 'log'.")
    if scan_max <= scan_min or (scan_scale == "log" and scan_min <= 0.0):
        raise ValueError(
            "QCD-score scan requires scan_min < scan_max and a positive log minimum."
        )
    if scan_scale == "log" and any(value <= 0.0 for value in candidate_thresholds):
        raise ValueError("Log-scale QCD-score candidate thresholds must be positive.")
    if scan_points < 2:
        raise ValueError("QCD-score scan_points must be at least 2.")
    if any(value < scan_min or value > scan_max for value in candidate_thresholds):
        raise ValueError("QCD-score candidate thresholds must lie within the scan range.")

    distribution_bins = int(scan_payload.get("distribution_bins", 60))
    raw_log_range = [
        float(value)
        for value in list(scan_payload.get("distribution_log10_range", [-6.0, 0.0]))
    ]
    if distribution_bins <= 0:
        raise ValueError("QCD-score distribution_bins must be positive.")
    if len(raw_log_range) != 2 or raw_log_range[0] >= raw_log_range[1]:
        raise ValueError(
            "QCD-score distribution_log10_range must contain two ascending values."
        )

    truth_names = {truth.name for truth in truths}
    groups: list[NnQcdScoreGroup] = []
    for entry in list(scan_payload.get("groups", [])):
        categories = [str(name) for name in list(entry.get("truth_categories", []))]
        unknown = sorted(set(categories) - truth_names)
        if unknown:
            raise ValueError(
                "Unknown QCD-score scan truth categories: " + ", ".join(unknown)
            )
        if not categories:
            raise ValueError("Each QCD-score scan group requires truth_categories.")
        groups.append(
            NnQcdScoreGroup(
                name=str(entry["name"]),
                label=str(entry.get("label", entry["name"])),
                truth_categories=categories,
                color=str(entry["color"]) if entry.get("color") else None,
            )
        )
    _unique_names([group.name for group in groups], "QCD-score scan group")
    required_groups = {"ttHcc", "qcd", "ttX"}
    missing_groups = sorted(required_groups - {group.name for group in groups})
    if enabled and missing_groups:
        raise ValueError(
            "Enabled qcd_score_scan is missing groups: " + ", ".join(missing_groups)
        )

    working_point_bins = int(scan_payload.get("working_point_bins", 25))
    raw_score_range = [
        float(value)
        for value in list(
            scan_payload.get("working_point_score_range", [1e-5, 1.0])
        )
    ]
    if working_point_bins <= 0:
        raise ValueError("QCD working-point distribution bins must be positive.")
    working_point_scale = str(scan_payload.get("working_point_scale", "log"))
    if working_point_scale not in {"linear", "log"}:
        raise ValueError("QCD working_point_scale must be 'linear' or 'log'.")
    if (
        len(raw_score_range) != 2
        or raw_score_range[0] >= raw_score_range[1]
        or (working_point_scale == "log" and raw_score_range[0] <= 0.0)
    ):
        raise ValueError(
            "QCD working-point score range must be ascending and positive for log scale."
        )
    raw_reference_threshold = scan_payload.get("reference_threshold", 0.001)
    reference_threshold = (
        None if raw_reference_threshold is None else float(raw_reference_threshold)
    )
    if reference_threshold is not None and not (
        raw_score_range[0] < reference_threshold < raw_score_range[1]
    ):
        raise ValueError(
            "QCD reference_threshold must lie strictly inside the working-point score range."
        )

    distribution_groups: list[NnQcdScoreGroup] = []
    for entry in list(scan_payload.get("distribution_groups", [])):
        categories = [str(name) for name in list(entry.get("truth_categories", []))]
        unknown = sorted(set(categories) - truth_names)
        if unknown:
            raise ValueError(
                "Unknown QCD working-point distribution truth categories: "
                + ", ".join(unknown)
            )
        if not categories:
            raise ValueError(
                "Each QCD working-point distribution group requires truth_categories."
            )
        distribution_groups.append(
            NnQcdScoreGroup(
                name=str(entry["name"]),
                label=str(entry.get("label", entry["name"])),
                truth_categories=categories,
                color=str(entry["color"]) if entry.get("color") else None,
            )
        )
    _unique_names(
        [group.name for group in distribution_groups],
        "QCD working-point distribution group",
    )
    required_distribution_groups = {
        "ttHcc",
        "tt_light",
        "tt_ge1c",
        "tt_ge1b",
        "qcd",
    }
    missing_distribution_groups = sorted(
        required_distribution_groups - {group.name for group in distribution_groups}
    )
    if enabled and missing_distribution_groups:
        raise ValueError(
            "Enabled qcd_score_scan is missing distribution groups: "
            + ", ".join(missing_distribution_groups)
        )

    significance_payload = dict(scan_payload.get("significance", {}))
    significance_enabled = bool(significance_payload.get("enabled", False))
    significance_metric = str(
        significance_payload.get("metric", "s_over_sqrt_s_plus_b")
    )
    if significance_metric != "s_over_sqrt_s_plus_b":
        raise ValueError(
            "qcd_score_scan.significance.metric must be "
            "'s_over_sqrt_s_plus_b'."
        )
    significance_signals = [
        str(name) for name in list(significance_payload.get("signals", []))
    ]
    unknown_significance_signals = sorted(
        set(significance_signals) - truth_names
    )
    if unknown_significance_signals:
        raise ValueError(
            "Unknown QCD-score significance signals: "
            + ", ".join(unknown_significance_signals)
        )
    if len(significance_signals) != len(set(significance_signals)):
        raise ValueError("QCD-score significance signals must be unique.")
    if enabled and significance_enabled and not significance_signals:
        raise ValueError(
            "Enabled qcd_score_scan significance requires non-empty signals."
        )
    return NnQcdScoreScanConfig(
        enabled=enabled,
        score_name=score.name,
        score_branch=score.branch,
        direction=direction,
        candidate_thresholds=candidate_thresholds,
        scan_min=scan_min,
        scan_max=scan_max,
        scan_points=scan_points,
        scan_scale=scan_scale,
        distribution_bins=distribution_bins,
        distribution_log10_range=(raw_log_range[0], raw_log_range[1]),
        groups=groups,
        working_point_bins=working_point_bins,
        working_point_score_range=(raw_score_range[0], raw_score_range[1]),
        reference_threshold=reference_threshold,
        working_point_scale=working_point_scale,
        distribution_groups=distribution_groups,
        significance_enabled=significance_enabled,
        significance_signals=significance_signals,
        significance_metric=significance_metric,
    )


def _load_weighting_diagnostics(
    payload: dict[str, Any],
    scores: list[NnScoreClass],
) -> NnWeightingDiagnosticsConfig:
    diagnostics = dict(payload.get("weighting_diagnostics", {}))
    enabled = bool(diagnostics.get("enabled", False))
    score_names = [score.name for score in scores]
    score_name_set = set(score_names)

    reweight_variables: dict[str, list[float]] = {}
    for name, raw_edges in dict(diagnostics.get("reweight_variables", {})).items():
        edges = [float(value) for value in list(raw_edges)]
        if len(edges) < 2 or any(right <= left for left, right in zip(edges, edges[1:])):
            raise ValueError(
                f"Weighting-diagnostics reweight variable '{name}' needs ascending bin edges."
            )
        reweight_variables[str(name)] = edges

    class_weights = {
        str(name): float(value)
        for name, value in dict(diagnostics.get("class_weights", {})).items()
    }
    unknown_weight_classes = sorted(set(class_weights) - score_name_set)
    if unknown_weight_classes:
        raise ValueError(
            "Unknown weighting-diagnostics class weights: "
            + ", ".join(unknown_weight_classes)
        )
    if any(value <= 0.0 for value in class_weights.values()):
        raise ValueError("Weighting-diagnostics class weights must be positive.")
    if enabled and set(class_weights) != score_name_set:
        missing = sorted(score_name_set - set(class_weights))
        raise ValueError(
            "Enabled weighting_diagnostics requires a class weight for every score class; "
            "missing: " + ", ".join(missing)
        )

    key_auc_pairs: list[tuple[str, str]] = []
    for raw_pair in list(diagnostics.get("key_auc_pairs", [])):
        pair = [str(name) for name in list(raw_pair)]
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError("Each weighting-diagnostics key AUC pair needs two distinct classes.")
        unknown = sorted(set(pair) - score_name_set)
        if unknown:
            raise ValueError(
                "Unknown weighting-diagnostics key AUC classes: " + ", ".join(unknown)
            )
        key_auc_pairs.append((pair[0], pair[1]))

    model_output_classes = [
        str(name)
        for name in list(diagnostics.get("model_output_classes", score_names))
    ]
    persisted_score_classes = [
        str(name)
        for name in list(diagnostics.get("persisted_score_classes", score_names))
    ]
    if persisted_score_classes != score_names:
        raise ValueError(
            "weighting_diagnostics.persisted_score_classes must match scores in order."
        )
    return NnWeightingDiagnosticsConfig(
        enabled=enabled,
        reweight_variables=reweight_variables,
        class_weights=class_weights,
        key_auc_pairs=key_auc_pairs,
        model_output_classes=model_output_classes,
        persisted_score_classes=persisted_score_classes,
    )


def load_nn_study_config(
    path: str | Path,
    repo_root: str | Path,
    *,
    outdir: str | None = None,
    selection: str | None = None,
    max_files_per_sample: int | None = None,
    discover_sample_files: bool = True,
) -> NnStudyConfig:
    config_path = Path(path).resolve()
    repo_root_path = Path(repo_root).resolve()
    payload = load_json_maybe_with_comments(config_path)
    study = dict(payload.get("study", {}))
    require_complete = bool(study.get("require_complete_samples", False))
    if discover_sample_files and require_complete:
        gen_sumw = payload.get("normalization", {}).get("gen_sumw_file")
        if gen_sumw is not None and not _resolve_path(gen_sumw, config_path.parent).is_file():
            raise FileNotFoundError(
                f"Production incomplete: gen_sumw file does not exist: {gen_sumw}. "
                "Wait for Pepper production to finish before preparing the final cache."
            )

    scores = [
        NnScoreClass(
            name=str(entry["name"]),
            label=str(entry.get("label", entry["name"])),
            branch=str(entry["branch"]),
            color=str(entry["color"]) if entry.get("color") else None,
        )
        for entry in list(payload.get("scores", []))
    ]
    auxiliary_scores = [
        NnScoreClass(
            name=str(entry["name"]),
            label=str(entry.get("label", entry["name"])),
            branch=str(entry["branch"]),
            color=str(entry["color"]) if entry.get("color") else None,
        )
        for entry in list(payload.get("auxiliary_scores", []))
    ]
    truths = [
        NnTruthCategory(
            name=str(entry["name"]),
            label=str(entry.get("label", entry["name"])),
            expression=str(entry["expression"]),
            color=str(entry["color"]) if entry.get("color") else None,
        )
        for entry in list(payload.get("truth_categories", []))
    ]
    raw_prediction_truths = payload.get("prediction_truth_categories")
    prediction_truths = (
        truths
        if raw_prediction_truths is None
        else [
            NnTruthCategory(
                name=str(entry["name"]),
                label=str(entry.get("label", entry["name"])),
                expression=str(entry["expression"]),
                color=str(entry["color"]) if entry.get("color") else None,
            )
            for entry in list(raw_prediction_truths)
        ]
    )
    diagnostic_truths = [
        NnTruthCategory(
            name=str(entry["name"]),
            label=str(entry.get("label", entry["name"])),
            expression=str(entry["expression"]),
            color=str(entry["color"]) if entry.get("color") else None,
        )
        for entry in list(payload.get("diagnostic_truth_categories", []))
    ]
    if not scores:
        raise ValueError("Config field 'scores' must be a non-empty list.")
    if not truths:
        raise ValueError("Config field 'truth_categories' must be a non-empty list.")
    all_scores = [*scores, *auxiliary_scores]
    _unique_names([item.name for item in all_scores], "score")
    _unique_names([item.branch for item in all_scores], "score branch")
    analysis_discriminants = [
        str(name) for name in list(payload.get("analysis_discriminants", []))
    ]
    unknown_discriminants = sorted(
        set(analysis_discriminants) - {score.name for score in all_scores}
    )
    if unknown_discriminants:
        raise ValueError(
            "Unknown analysis discriminants: " + ", ".join(unknown_discriminants)
        )
    _unique_names(analysis_discriminants, "analysis discriminant")
    _unique_names([item.name for item in truths], "truth category")
    _unique_names(
        [item.name for item in prediction_truths], "prediction truth category"
    )
    _unique_names(
        [item.name for item in diagnostic_truths], "diagnostic truth category"
    )
    if [item.name for item in scores] != [item.name for item in prediction_truths]:
        raise ValueError(
            "Score classes and prediction truth categories must use the same names "
            "in the same order."
        )
    input_location = str(study.get("input_location", ""))
    raw_sample_file_pattern = study.get("sample_file_pattern")
    sample_file_pattern = (
        None if raw_sample_file_pattern is None else str(raw_sample_file_pattern)
    )
    samples: list[NnSample] = []
    for entry in list(payload.get("samples", [])):
        name = str(entry["name"])
        dataset = str(entry.get("dataset", name))
        if sample_file_pattern is not None:
            try:
                raw_patterns = [
                    sample_file_pattern.format(
                        input_location=input_location.rstrip("/"),
                        name=name,
                        dataset=dataset,
                    )
                ]
            except KeyError as exc:
                raise ValueError(
                    "study.sample_file_pattern supports only {input_location}, "
                    "{name}, and {dataset}."
                ) from exc
        else:
            if "files" not in entry:
                raise ValueError(
                    f"Sample '{name}' needs files when study.sample_file_pattern is unset."
                )
            raw_patterns = [str(pattern) for pattern in list(entry["files"])]
        patterns = [
            str(_resolve_path(pattern, config_path.parent))
            for pattern in raw_patterns
        ]
        files = expand_file_patterns(patterns) if discover_sample_files else []
        samples.append(
            NnSample(
                name=name,
                dataset=dataset,
                label=str(entry.get("label", name)),
                files=files,
                selection=str(entry.get("selection", "")),
            )
        )
    if not samples:
        raise ValueError("Config field 'samples' must be a non-empty list.")
    _unique_names([sample.name for sample in samples], "sample")
    if discover_sample_files and require_complete:
        missing = [sample.name for sample in samples if not sample.files]
        if missing:
            raise FileNotFoundError(
                "Production incomplete: no ROOT files for configured samples: "
                + ", ".join(missing)
                + ". Wait for Pepper production to finish before preparing the final cache."
            )
    mass_sculpting = _load_mass_sculpting(payload, all_scores, truths, samples)
    score_significance = _load_score_significance(payload, all_scores, truths)
    score_significance_2d = _load_score_significance_2d(
        payload, all_scores, truths
    )
    score_significance_2d_hierarchical = _load_score_significance_2d(
        payload,
        all_scores,
        truths,
        config_key="score_significance_2d_hierarchical",
    )
    significance_mass_window = _load_significance_mass_window(payload)
    analysis_regions = _load_analysis_regions(payload, all_scores)
    qcd_score_scan = _load_qcd_score_scan(payload, all_scores, truths)
    weighting_diagnostics = _load_weighting_diagnostics(payload, scores)

    normalization = dict(payload.get("normalization", {}))
    gen_sumw_file = normalization.get("gen_sumw_file")
    xsec_file = normalization.get("xsec_file")
    lumi_fb = normalization.get("lumi_fb")
    if gen_sumw_file is None or xsec_file is None or lumi_fb is None:
        raise ValueError("normalization.gen_sumw_file, xsec_file, and lumi_fb are required.")

    effective_max_files = max_files_per_sample
    if effective_max_files is None and study.get("max_files_per_sample") is not None:
        effective_max_files = int(study["max_files_per_sample"])
    if effective_max_files is not None and effective_max_files <= 0:
        raise ValueError("max_files_per_sample must be positive.")

    return NnStudyConfig(
        config_path=config_path,
        repo_root=repo_root_path,
        channel=str(study.get("channel", "unknown")),
        input_location=input_location,
        sample_file_pattern=sample_file_pattern,
        outdir=_resolve_outdir(repo_root_path, study.get("outdir"), outdir),
        tree_name=str(study.get("tree_name", DEFAULT_TREE_NAME)),
        weight_branch=str(study.get("weight_branch", DEFAULT_WEIGHT_BRANCH)),
        selection=str(study.get("selection", "") if selection is None else selection),
        selection_branches=[str(name) for name in list(study.get("selection_branches", []))],
        truth_branches=[str(name) for name in list(study.get("truth_branches", []))],
        analysis_branches=[str(name) for name in list(study.get("analysis_branches", []))],
        flatten_first_branches=[
            str(name) for name in list(study.get("flatten_first_branches", []))
        ],
        scores=scores,
        auxiliary_scores=auxiliary_scores,
        analysis_discriminants=analysis_discriminants,
        truth_categories=truths,
        prediction_truth_categories=prediction_truths,
        diagnostic_truth_categories=diagnostic_truths,
        samples=samples,
        gen_sumw_file=str(_resolve_path(str(gen_sumw_file), config_path.parent)),
        xsec_file=str(_resolve_path(str(xsec_file), config_path.parent)),
        lumi_fb=float(lumi_fb),
        uproot_step_size=str(study.get("uproot_step_size", DEFAULT_UPROOT_STEP_SIZE)),
        max_files_per_sample=effective_max_files,
        plot_options=dict(payload.get("plot", {})),
        regressed_mass_fine_binning=dict(payload.get("regressed_mass_fine_binning", {})),
        qcd_data_mc=dict(payload.get("qcd_data_mc", {})),
        tthz_qcd_discriminant_enabled=bool(
            payload.get("tthz_qcd_discriminant", {}).get("enabled", True)
        ),
        mass_sculpting=mass_sculpting,
        score_significance=score_significance,
        score_significance_2d=score_significance_2d,
        score_significance_2d_hierarchical=score_significance_2d_hierarchical,
        significance_mass_window=significance_mass_window,
        analysis_regions=analysis_regions,
        qcd_score_scan=qcd_score_scan,
        weighting_diagnostics=weighting_diagnostics,
        validate_auc_with_sklearn=bool(study.get("validate_auc_with_sklearn", True)),
        year=str(study.get("year", "2024")),
        energy_tev=float(study.get("energy_tev", 13.6)),
    )
