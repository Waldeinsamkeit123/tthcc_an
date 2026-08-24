from __future__ import annotations

import ast
from dataclasses import dataclass
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
    distribution_bins: int
    distribution_log10_range: tuple[float, float]
    groups: list[NnQcdScoreGroup]
    working_point_bins: int
    working_point_score_range: tuple[float, float]
    reference_threshold: float
    distribution_groups: list[NnQcdScoreGroup]


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
    truth_categories: list[NnTruthCategory]
    samples: list[NnSample]
    gen_sumw_file: str
    xsec_file: str
    lumi_fb: float
    uproot_step_size: str
    max_files_per_sample: int | None
    plot_options: dict[str, Any]
    mass_sculpting: NnMassSculptingConfig
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
    scans: list[NnMassScan] = []
    for entry in list(mass_payload.get("scans", [])):
        reference = str(entry["score"])
        if reference not in score_by_reference:
            raise ValueError(f"Unknown mass-sculpting score '{reference}'.")
        score = score_by_reference[reference]
        direction = str(entry["direction"])
        if direction not in {">", ">=", "<", "<="}:
            raise ValueError("Mass-sculpting direction must be one of >, >=, <, <=.")
        thresholds = [float(value) for value in list(entry.get("thresholds", []))]
        if not thresholds:
            raise ValueError(f"Mass-sculpting scan '{reference}' has no thresholds.")
        if thresholds != sorted(thresholds):
            raise ValueError(f"Mass-sculpting thresholds for '{reference}' must be ascending.")
        scans.append(
            NnMassScan(
                score_name=score.name,
                score_branch=score.branch,
                direction=direction,
                thresholds=thresholds,
            )
        )

    enabled = bool(mass_payload.get("enabled", False))
    if enabled and (not populations or not variables or not scans):
        raise ValueError(
            "Enabled mass_sculpting requires non-empty populations, variables, and scans."
        )
    return NnMassSculptingConfig(
        enabled=enabled,
        populations=populations,
        variables=variables,
        scans=scans,
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
    if enabled and not candidate_thresholds:
        raise ValueError("Enabled qcd_score_scan requires candidate_thresholds.")
    if candidate_thresholds != sorted(candidate_thresholds):
        raise ValueError("QCD-score candidate thresholds must be ascending.")
    if len(candidate_thresholds) != len(set(candidate_thresholds)):
        raise ValueError("QCD-score candidate thresholds must be unique.")
    if any(value <= 0.0 for value in candidate_thresholds):
        raise ValueError("QCD-score candidate thresholds must be positive.")

    scan_min = float(scan_payload.get("scan_min", 1e-6))
    scan_max = float(scan_payload.get("scan_max", 1.0))
    scan_points = int(scan_payload.get("scan_points", 100))
    if scan_min <= 0.0 or scan_max <= scan_min:
        raise ValueError("QCD-score scan requires 0 < scan_min < scan_max.")
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
    if (
        len(raw_score_range) != 2
        or raw_score_range[0] <= 0.0
        or raw_score_range[0] >= raw_score_range[1]
    ):
        raise ValueError(
            "QCD working-point score range must contain two ascending positive values."
        )
    reference_threshold = float(scan_payload.get("reference_threshold", 0.001))
    if not raw_score_range[0] < reference_threshold < raw_score_range[1]:
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
    return NnQcdScoreScanConfig(
        enabled=enabled,
        score_name=score.name,
        score_branch=score.branch,
        direction=direction,
        candidate_thresholds=candidate_thresholds,
        scan_min=scan_min,
        scan_max=scan_max,
        scan_points=scan_points,
        distribution_bins=distribution_bins,
        distribution_log10_range=(raw_log_range[0], raw_log_range[1]),
        groups=groups,
        working_point_bins=working_point_bins,
        working_point_score_range=(raw_score_range[0], raw_score_range[1]),
        reference_threshold=reference_threshold,
        distribution_groups=distribution_groups,
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
) -> NnStudyConfig:
    config_path = Path(path).resolve()
    repo_root_path = Path(repo_root).resolve()
    payload = load_json_maybe_with_comments(config_path)
    study = dict(payload.get("study", {}))

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
    if not scores:
        raise ValueError("Config field 'scores' must be a non-empty list.")
    if not truths:
        raise ValueError("Config field 'truth_categories' must be a non-empty list.")
    all_scores = [*scores, *auxiliary_scores]
    _unique_names([item.name for item in all_scores], "score")
    _unique_names([item.branch for item in all_scores], "score branch")
    _unique_names([item.name for item in truths], "truth category")
    if [item.name for item in scores] != [item.name for item in truths]:
        raise ValueError("Score classes and truth categories must use the same names in the same order.")
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
        files = expand_file_patterns(patterns)
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
    mass_sculpting = _load_mass_sculpting(payload, all_scores, truths, samples)
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
        truth_categories=truths,
        samples=samples,
        gen_sumw_file=str(_resolve_path(str(gen_sumw_file), config_path.parent)),
        xsec_file=str(_resolve_path(str(xsec_file), config_path.parent)),
        lumi_fb=float(lumi_fb),
        uproot_step_size=str(study.get("uproot_step_size", DEFAULT_UPROOT_STEP_SIZE)),
        max_files_per_sample=effective_max_files,
        plot_options=dict(payload.get("plot", {})),
        mass_sculpting=mass_sculpting,
        qcd_score_scan=qcd_score_scan,
        weighting_diagnostics=weighting_diagnostics,
        validate_auc_with_sklearn=bool(study.get("validate_auc_with_sklearn", True)),
        year=str(study.get("year", "2024")),
        energy_tev=float(study.get("energy_tev", 13.6)),
    )
