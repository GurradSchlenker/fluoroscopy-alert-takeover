"""Shared command-line plumbing.

Ref: none - release-internal. Every command takes one experiment config plus
``key=value`` overrides, writes its record under the run directory and sets the process
exit status from what it actually did.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from ..analysis.pipeline import AnalysisConfig
from ..cohort.builder import build_person_windows
from ..cohort.layout import CohortLayout
from ..falsification.bias_factor import DEFAULT_GAMMA_GRID
from ..forest.causal_forest import ForestConfig
from ..operating_point.abstention import AbstentionBand
from ..overlap.gate import DEFAULT_GRID as DEFAULT_GATE_GRID
from ..overlap.gate import DEFAULT_TOLERANCE as DEFAULT_GATE_TOLERANCE
from ..protocol.schema import PersonWindowTable
from ..protocol.windowing import WindowSpec
from ..utils.config import Mapping, compose, dig, load_yaml
from ..utils.io import write_json
from ..utils.logging import configure_logging, get_logger
from ..utils.seeding import set_seed

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BLOCKED = 2

DEFAULT_CONFIG_DIR = Path("configs")
_LOG = get_logger("cli")


@dataclass(frozen=True)
class CommandContext:
    """Resolved configuration of one command invocation."""

    name: str
    config: Mapping
    run_dir: Path
    seed: int

    def record(self, payload: Mapping) -> Path:
        """Write a stage record into the run directory and return its path."""
        return write_json(self.run_dir / f"{self.name}.json", payload)


def _parse_float_tuple(value: object, fallback: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return fallback
    if not isinstance(value, list):
        raise TypeError("expected a list of numbers in the config")
    return tuple(float(item) for item in value)


def build_parser(description: str) -> argparse.ArgumentParser:
    """Parser shared by every command."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--experiment",
        required=True,
        help="path to the experiment config, for example configs/experiment/primary.yaml",
    )
    parser.add_argument("--root", default=None, help="directory holding the release config tree")
    parser.add_argument("--run-dir", default=None, help="directory for this run's records")
    parser.add_argument("--cohort", default=None, help="override the cohort root path")
    parser.add_argument(
        "overrides",
        nargs="*",
        default=[],
        help="key=value overrides applied after the config is composed",
    )
    return parser


def prepare_context(name: str, description: str, argv: list[str] | None = None) -> CommandContext:
    """Compose the configuration chain, install the seed and log the invocation."""
    parser = build_parser(description)
    args = parser.parse_args(argv)
    root = Path(args.root) if args.root else DEFAULT_CONFIG_DIR
    experiment = Path(args.experiment)
    if not experiment.is_absolute():
        experiment = Path(experiment)
    chain: list[Path] = [
        root / "protocol" / "default.yaml",
        root / "cohort" / "default.yaml",
        root / "alert" / "default.yaml",
        root / "gate" / "default.yaml",
        root / "estimator" / "default.yaml",
        root / "mediation" / "default.yaml",
        root / "falsification" / "default.yaml",
        root / "operating_point" / "default.yaml",
        root / "strata" / "default.yaml",
        root / "inference" / "default.yaml",
        experiment,
    ]
    existing = [path for path in chain if path.is_file()]
    if not existing:
        raise FileNotFoundError("no configuration files were found")
    overrides = list(args.overrides)
    if args.cohort:
        overrides.append(f"cohort.root={args.cohort}")
    config = compose(existing, overrides)
    seed = int(dig(config, "protocol.seed", 20260101))
    set_seed(seed)
    configure_logging(str(dig(config, "protocol.log_level", "INFO")))
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else Path(str(dig(config, "protocol.run_dir", "runs/primary")))
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    _LOG.info(
        "command %s: %d config files composed, run directory %s", name, len(existing), run_dir
    )
    return CommandContext(name=name, config=config, run_dir=run_dir, seed=seed)


def analysis_config_from(config: Mapping) -> AnalysisConfig:
    """Build the pipeline configuration from a composed mapping."""
    gate = dig(config, "gate", {}) or {}
    estimator = dig(config, "estimator", {}) or {}
    operating = dig(config, "operating_point", {}) or {}
    strata = dig(config, "strata", {}) or {}
    inference = dig(config, "inference", {}) or {}
    protocol = dig(config, "protocol", {}) or {}
    return AnalysisConfig(
        n_folds=int(estimator.get("n_folds", 5)),
        clip_gamma=float(estimator.get("clip_gamma", 0.01)),
        outcome_learner=str(estimator.get("outcome_learner", "glm")),
        gate_tolerance=float(gate.get("tolerance", DEFAULT_GATE_TOLERANCE)),
        gate_grid=_parse_float_tuple(gate.get("grid"), DEFAULT_GATE_GRID),
        seed=int(protocol.get("seed", 20260101)),
        n_bootstrap=int(inference.get("bootstrap_replicates", 1000)),
        bootstrap_seed=int(inference.get("bootstrap_seed", 20260102)),
        primary_outcome=str(protocol.get("primary_outcome", "takeover")),
        secondary_outcome=str(protocol.get("secondary_outcome", "completion")),
        mediator=str(protocol.get("mediator", "hand_back")),
        consistency_floor=float(protocol.get("consistency_floor", 0.90)),
        gate_ess_floor=float(gate.get("ess_floor", 0.20)),
        alpha=float(strata.get("alpha", 0.05)),
        abstention=AbstentionBand(
            float(operating.get("abstention_low", 0.30)),
            float(operating.get("abstention_high", 0.40)),
        ),
        thresholds=_parse_float_tuple(operating.get("thresholds"), (0.15, 0.25, 0.35, 0.45, 0.55)),
        window_multipliers=_parse_float_tuple(
            dig(config, "mediation.multipliers"), (1.0, 2.0, 3.0, 4.0)
        ),
        forest=ForestConfig(
            trees=int(strata.get("forest_trees", 200)),
            min_leaf=int(strata.get("forest_min_leaf", 50)),
            max_depth=int(strata.get("forest_max_depth", 4)),
            seed=int(protocol.get("seed", 20260101)),
        ),
    )


def gamma_grid_from(config: Mapping) -> tuple[float, ...]:
    """Bias-factor grid of the sensitivity analysis."""
    grid = dig(config, "falsification.gamma_grid")
    return _parse_float_tuple(grid, DEFAULT_GAMMA_GRID)


def cohort_root_from(config: Mapping) -> Path:
    """Cohort root of a composed config."""
    return Path(str(dig(config, "cohort.root", "data/cohort")))


def window_spec_from(config: Mapping) -> WindowSpec:
    """Analysis-period layout of a composed config."""
    return WindowSpec(
        length_s=float(dig(config, "protocol.window_length_s", 30.0)),
        stride_s=float(dig(config, "protocol.window_stride_s", 15.0)),
        response_s=float(dig(config, "protocol.response_window_s", 30.0)),
    )


def load_table(config: Mapping) -> PersonWindowTable:
    """Build the person-window table of a composed config.

    Raises :class:`FileNotFoundError` when the cohort root is not staged, so every command
    that needs the cohort reports a blocked run rather than an empty analysis.
    """

    root = cohort_root_from(config)
    layout = CohortLayout(root=root)
    absent = layout.missing()
    if absent:
        raise FileNotFoundError("cohort root is not staged; missing: " + ", ".join(absent))
    return build_person_windows(layout, window_spec_from(config))


__all__ = [
    "EXIT_BLOCKED",
    "EXIT_FAILED",
    "EXIT_OK",
    "CommandContext",
    "analysis_config_from",
    "cohort_root_from",
    "gamma_grid_from",
    "load_table",
    "load_yaml",
    "prepare_context",
    "window_spec_from",
]
