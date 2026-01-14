#!/usr/bin/env python3
"""Behavior search runner for ABM optimization.

Performs random search over parameter space to find configurations that
minimize distance to target enterprise rates.

Supports both uncalibrated and calibrated synthetic data, and can load
target enterprise rates from LSMS derived data (FLAG 6 remediation).

Usage:
    # Uncalibrated (legacy)
    python3 scripts/run_behavior_search.py
    python3 scripts/run_behavior_search.py --n-candidates 50 --seeds 2

    # Calibrated with LSMS targets (FLAG 1 + FLAG 6 remediation)
    python3 scripts/run_behavior_search.py \
        --calibration artifacts/calibration/tanzania/calibration.json \
        --targets-from-lsms
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from abm_enterprise.data.schemas import SimulationConfig
from abm_enterprise.data.synthetic import (
    generate_synthetic_households,
    generate_synthetic_panel_from_file,
)
from abm_enterprise.model import EnterpriseCopingModel
from abm_enterprise.policies.rule import RulePolicy
from abm_enterprise.utils.rng import set_seed


def load_lsms_enterprise_rates(country: str = "tanzania") -> dict[int, float]:
    """Load enterprise rates by wave from LSMS derived data.

    Args:
        country: Country code (tanzania or ethiopia)

    Returns:
        Dictionary mapping wave to enterprise rate
    """
    targets_path = Path(f"data/processed/{country}/derived/household_targets.parquet")
    if not targets_path.exists():
        raise FileNotFoundError(f"LSMS targets not found: {targets_path}")

    df = pd.read_parquet(targets_path)
    rates = df.groupby("wave")["enterprise_indicator"].mean().to_dict()
    return {int(k): float(v) for k, v in rates.items()}


@dataclass
class SearchConfig:
    """Configuration for behavior search."""

    # Target enterprise rates by wave (from LSMS or hardcoded)
    target_enterprise_rates: dict[int, float]

    # Parameter bounds
    price_threshold_range: tuple[float, float] = (-0.4, 0.1)
    asset_threshold_range: tuple[float, float] = (-1.5, 1.5)
    exit_threshold_range: tuple[float, float] = (-2.0, 0.0)

    # Search parameters
    n_candidates: int = 40
    seeds_per_candidate: int = 2
    num_households: int = 100
    num_waves: int = 4

    # Calibration (optional)
    calibration_path: str | None = None

    # Data provenance (per DATA_CONTRACT.md)
    data_source: str = "synthetic_uncalibrated"  # synthetic_uncalibrated | calibrated | lsms_derived
    target_source: str = "hardcoded"  # hardcoded | lsms_derived

    @classmethod
    def default(cls, use_lsms_targets: bool = False, calibration_path: str | None = None) -> "SearchConfig":
        """Create default search configuration.

        Args:
            use_lsms_targets: If True, load targets from LSMS data (FLAG 6 fix)
            calibration_path: Path to calibration.json for calibrated synthetic data (FLAG 1 fix)
        """
        if use_lsms_targets:
            target_rates = load_lsms_enterprise_rates("tanzania")
            target_source = "lsms_derived"
        else:
            # Legacy hardcoded values
            target_rates = {
                1: 0.25,  # Wave 1
                2: 0.28,  # Wave 2
                3: 0.32,  # Wave 3
                4: 0.35,  # Wave 4
            }
            target_source = "hardcoded"

        data_source = "calibrated" if calibration_path else "synthetic_uncalibrated"

        return cls(
            target_enterprise_rates=target_rates,
            calibration_path=calibration_path,
            data_source=data_source,
            target_source=target_source,
        )


@dataclass
class SearchCandidate:
    """A candidate solution in the search space."""

    candidate_id: int
    price_threshold: float
    asset_threshold: float
    exit_threshold: float

    # Results (filled after evaluation)
    mean_enterprise_rate: float = 0.0
    enterprise_rate_by_wave: dict | None = None
    objective: float = float("inf")
    objective_std: float = 0.0


def compute_objective(
    simulated_rates: dict[int, float],
    target_rates: dict[int, float],
) -> float:
    """Compute objective function (MSE between simulated and target rates)."""
    squared_errors = []
    for wave in target_rates:
        if wave in simulated_rates:
            error = simulated_rates[wave] - target_rates[wave]
            squared_errors.append(error ** 2)
    return np.mean(squared_errors) if squared_errors else float("inf")


def evaluate_candidate(
    candidate: SearchCandidate,
    config: SearchConfig,
    seeds: list[int],
) -> list[float]:
    """Evaluate a candidate across multiple seeds, return list of objectives."""
    objectives = []
    all_wave_rates = {w: [] for w in range(1, config.num_waves + 1)}

    for seed in seeds:
        set_seed(seed)

        # Generate synthetic data
        if config.calibration_path:
            # Use calibrated synthetic generation
            household_data = generate_synthetic_panel_from_file(
                calibration_path=config.calibration_path,
                n_households=config.num_households,
                seed=seed,
            )
        else:
            # Use legacy uncalibrated generation
            household_data = generate_synthetic_households(
                n=config.num_households,
                waves=config.num_waves,
                country="tanzania",
            )

        # Create policy with candidate parameters
        policy = RulePolicy(
            price_threshold=candidate.price_threshold,
            asset_threshold=candidate.asset_threshold,
            exit_asset_threshold=candidate.exit_threshold,
        )

        # Create config
        sim_config = SimulationConfig(
            country="tanzania",
            scenario="search",
            seed=seed,
            num_waves=config.num_waves,
        )

        # Run model
        model = EnterpriseCopingModel(
            config=sim_config,
            household_data=household_data,
            policy=policy,
        )
        model.run()

        # Get outcomes
        outcomes = model.get_outcomes_dataframe()

        # Compute enterprise rate by wave
        wave_rates = outcomes.groupby("wave")["enterprise_status"].mean().to_dict()

        for w, rate in wave_rates.items():
            all_wave_rates[w].append(rate)

        # Compute objective for this seed
        obj = compute_objective(wave_rates, config.target_enterprise_rates)
        objectives.append(obj)

    # Store mean rates
    candidate.enterprise_rate_by_wave = {
        w: np.mean(rates) for w, rates in all_wave_rates.items() if rates
    }
    candidate.mean_enterprise_rate = np.mean(
        list(candidate.enterprise_rate_by_wave.values())
    )

    return objectives


def random_search(
    config: SearchConfig,
    output_dir: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """Perform random search and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Search configuration:")
        print(f"  Data source: {config.data_source}")
        print(f"  Target source: {config.target_source}")
        if config.calibration_path:
            print(f"  Calibration: {config.calibration_path}")
        print(f"  Target rates:")
        for wave, rate in sorted(config.target_enterprise_rates.items()):
            print(f"    Wave {wave}: {rate:.1%}")
        print(f"  N candidates: {config.n_candidates}")
        print(f"  Seeds per candidate: {config.seeds_per_candidate}")
        print()

    rng = np.random.default_rng(seed=42)

    candidates = []
    results = []

    # Generate candidates
    for i in range(config.n_candidates):
        candidate = SearchCandidate(
            candidate_id=i,
            price_threshold=rng.uniform(*config.price_threshold_range),
            asset_threshold=rng.uniform(*config.asset_threshold_range),
            exit_threshold=rng.uniform(*config.exit_threshold_range),
        )
        candidates.append(candidate)

    # Evaluate candidates
    for i, candidate in enumerate(candidates):
        if verbose:
            print(
                f"[{i+1}/{config.n_candidates}] "
                f"pt={candidate.price_threshold:.3f}, "
                f"at={candidate.asset_threshold:.3f}, "
                f"et={candidate.exit_threshold:.3f}...",
                end="",
                flush=True,
            )

        seeds = list(range(42, 42 + config.seeds_per_candidate))
        objectives = evaluate_candidate(candidate, config, seeds)

        candidate.objective = np.mean(objectives)
        candidate.objective_std = np.std(objectives)

        if verbose:
            print(f" obj={candidate.objective:.4f}")

        # Store results
        result = {
            "candidate_id": candidate.candidate_id,
            "price_threshold": candidate.price_threshold,
            "asset_threshold": candidate.asset_threshold,
            "exit_threshold": candidate.exit_threshold,
            "mean_enterprise_rate": candidate.mean_enterprise_rate,
            "objective": candidate.objective,
            "objective_std": candidate.objective_std,
        }
        # Add wave-by-wave rates
        if candidate.enterprise_rate_by_wave:
            for w, rate in candidate.enterprise_rate_by_wave.items():
                result[f"rate_wave_{w}"] = rate
        results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(results).sort_values("objective")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Candidates
    candidates_path = output_dir / f"candidates_{timestamp}.parquet"
    results_df.to_parquet(candidates_path, index=False)

    csv_path = output_dir / f"candidates_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)

    # Config
    config_path = output_dir / f"search_config_{timestamp}.json"
    with open(config_path, "w") as f:
        config_dict = asdict(config)
        json.dump(config_dict, f, indent=2)

    # Create symlinks to latest
    for name, path in [
        ("candidates_latest.parquet", candidates_path),
        ("candidates_latest.csv", csv_path),
        ("search_config_latest.json", config_path),
    ]:
        link = output_dir / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(path.name)

    if verbose:
        print(f"\nSearch complete!")
        print(f"  Candidates: {candidates_path}")
        print(f"  CSV: {csv_path}")
        print(f"  Config: {config_path}")

        print("\nTop 5 candidates:")
        print(results_df.head()[
            ["candidate_id", "price_threshold", "asset_threshold",
             "exit_threshold", "objective"]
        ].to_string(index=False))

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run ABM behavior search")
    parser.add_argument(
        "--n-candidates", type=int, default=40,
        help="Number of candidates to evaluate (default: 40)"
    )
    parser.add_argument(
        "--seeds", type=int, default=2,
        help="Seeds per candidate (default: 2)"
    )
    parser.add_argument(
        "--households", type=int, default=100,
        help="Households per simulation (default: 100)"
    )
    parser.add_argument(
        "--calibration", type=Path, default=None,
        help="Path to calibration.json for calibrated synthetic data"
    )
    parser.add_argument(
        "--targets-from-lsms", action="store_true",
        help="Load target enterprise rates from LSMS derived data (FLAG 6 fix)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: outputs/search/uncalibrated or outputs/search/calibrated)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Determine output directory based on calibration
    if args.output_dir:
        output_dir = args.output_dir
    elif args.calibration:
        output_dir = Path("outputs/search/calibrated")
    else:
        output_dir = Path("outputs/search/uncalibrated")

    config = SearchConfig.default(
        use_lsms_targets=args.targets_from_lsms,
        calibration_path=str(args.calibration) if args.calibration else None,
    )
    config.n_candidates = args.n_candidates
    config.seeds_per_candidate = args.seeds
    config.num_households = args.households

    random_search(config, output_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()
