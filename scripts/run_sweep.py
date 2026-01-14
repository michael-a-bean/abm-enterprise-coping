#!/usr/bin/env python3
"""Parameter sweep runner for ABM sensitivity analysis.

Runs simulations across a grid of price_threshold and asset_threshold values,
collecting summary statistics for heatmap generation.

Supports both uncalibrated and calibrated synthetic data generation.

Usage:
    # Uncalibrated synthetic (exploratory)
    python scripts/run_sweep.py
    python scripts/run_sweep.py --grid-size 4 --seeds 2

    # Calibrated synthetic (FLAG 1 remediation)
    python scripts/run_sweep.py --calibration artifacts/calibration/tanzania/calibration.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterator
import itertools

import pandas as pd

from abm_enterprise.data.schemas import SimulationConfig
from abm_enterprise.data.synthetic import (
    generate_synthetic_households,
    generate_synthetic_panel_from_file,
)
from abm_enterprise.model import EnterpriseCopingModel
from abm_enterprise.policies.rule import RulePolicy
from abm_enterprise.utils.rng import set_seed


@dataclass
class SweepConfig:
    """Configuration for parameter sweep."""

    # Grid parameters
    price_thresholds: list[float]
    asset_thresholds: list[float]
    seeds: list[int]

    # Simulation parameters
    num_households: int = 100
    num_waves: int = 4
    country: str = "tanzania"

    # Calibration (optional)
    calibration_path: str | None = None  # Path to calibration.json

    # Data provenance (per DATA_CONTRACT.md)
    data_source: str = "synthetic_uncalibrated"  # synthetic_uncalibrated | calibrated | lsms_derived

    @classmethod
    def default_grid(
        cls,
        grid_size: int = 6,
        num_seeds: int = 2,
        calibration_path: str | None = None,
    ) -> "SweepConfig":
        """Create default sweep configuration."""
        import numpy as np

        return cls(
            price_thresholds=np.linspace(-0.3, 0.0, grid_size).tolist(),
            asset_thresholds=np.linspace(-1.0, 1.0, grid_size).tolist(),
            seeds=list(range(42, 42 + num_seeds)),
            calibration_path=calibration_path,
            data_source="calibrated" if calibration_path else "synthetic_uncalibrated",
        )


@dataclass
class SweepResult:
    """Result from a single sweep point."""

    price_threshold: float
    asset_threshold: float
    seed: int

    # Outcomes
    enterprise_rate: float
    mean_price_exposure: float
    mean_assets: float
    n_stayers: int
    n_copers: int
    n_none: int
    entry_rate: float
    exit_rate: float


def run_single_simulation(
    price_threshold: float,
    asset_threshold: float,
    seed: int,
    num_households: int = 100,
    num_waves: int = 4,
    country: str = "tanzania",
    calibration_path: str | None = None,
) -> SweepResult:
    """Run a single simulation with given parameters."""
    set_seed(seed)

    # Generate synthetic data
    if calibration_path:
        # Use calibrated synthetic generation
        household_data = generate_synthetic_panel_from_file(
            calibration_path=calibration_path,
            n_households=num_households,
            seed=seed,
        )
    else:
        # Use legacy uncalibrated generation
        household_data = generate_synthetic_households(
            n=num_households,
            waves=num_waves,
            country=country,
        )

    # Create policy with specified thresholds
    policy = RulePolicy(
        price_threshold=price_threshold,
        asset_threshold=asset_threshold,
    )

    # Create config
    config = SimulationConfig(
        country=country,
        scenario="sweep",
        seed=seed,
        num_waves=num_waves,
    )

    # Run model
    model = EnterpriseCopingModel(
        config=config,
        household_data=household_data,
        policy=policy,
    )
    model.run()

    # Get outcomes
    outcomes = model.get_outcomes_dataframe()

    # Compute summary statistics
    enterprise_rate = outcomes["enterprise_status"].mean()
    mean_price_exposure = outcomes["price_exposure"].mean()
    mean_assets = outcomes["assets_index"].mean()

    # Classification counts
    if "classification" in outcomes.columns:
        class_counts = outcomes.groupby("classification")["household_id"].nunique()
        n_stayers = class_counts.get("stayer", 0)
        n_copers = class_counts.get("coper", 0)
        n_none = class_counts.get("none", 0)
    else:
        n_stayers = n_copers = n_none = 0

    # Transition rates
    transitions = (
        outcomes
        .sort_values(["household_id", "wave"])
        .groupby("household_id")
        .apply(lambda g: g["enterprise_status"].diff().dropna())
    )
    if len(transitions) > 0:
        entry_rate = (transitions == 1).mean()
        exit_rate = (transitions == -1).mean()
    else:
        entry_rate = exit_rate = 0.0

    return SweepResult(
        price_threshold=price_threshold,
        asset_threshold=asset_threshold,
        seed=seed,
        enterprise_rate=enterprise_rate,
        mean_price_exposure=mean_price_exposure,
        mean_assets=mean_assets,
        n_stayers=int(n_stayers),
        n_copers=int(n_copers),
        n_none=int(n_none),
        entry_rate=float(entry_rate),
        exit_rate=float(exit_rate),
    )


def generate_sweep_points(config: SweepConfig) -> Iterator[tuple[float, float, int]]:
    """Generate all sweep points (price, asset, seed)."""
    for pt, at, seed in itertools.product(
        config.price_thresholds,
        config.asset_thresholds,
        config.seeds,
    ):
        yield pt, at, seed


def run_sweep(config: SweepConfig, output_dir: Path, verbose: bool = True) -> pd.DataFrame:
    """Run full parameter sweep and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_points = (
        len(config.price_thresholds) *
        len(config.asset_thresholds) *
        len(config.seeds)
    )

    if verbose:
        print(f"Running sweep: {total_points} points")
        print(f"  Data source: {config.data_source}")
        if config.calibration_path:
            print(f"  Calibration: {config.calibration_path}")
        print(f"  Price thresholds: {config.price_thresholds}")
        print(f"  Asset thresholds: {config.asset_thresholds}")
        print(f"  Seeds: {config.seeds}")

    for i, (pt, at, seed) in enumerate(generate_sweep_points(config)):
        if verbose:
            print(f"  [{i+1}/{total_points}] pt={pt:.2f}, at={at:.2f}, seed={seed}...", end="", flush=True)

        result = run_single_simulation(
            price_threshold=pt,
            asset_threshold=at,
            seed=seed,
            num_households=config.num_households,
            num_waves=config.num_waves,
            country=config.country,
            calibration_path=config.calibration_path,
        )
        results.append(asdict(result))

        if verbose:
            print(f" ent_rate={result.enterprise_rate:.1%}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Compute aggregates across seeds
    agg_df = (
        results_df
        .groupby(["price_threshold", "asset_threshold"])
        .agg({
            "enterprise_rate": ["mean", "std"],
            "entry_rate": ["mean", "std"],
            "exit_rate": ["mean", "std"],
            "n_stayers": "mean",
            "n_copers": "mean",
            "n_none": "mean",
        })
        .reset_index()
    )
    # Flatten column names
    agg_df.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in agg_df.columns
    ]

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Full results
    full_path = output_dir / f"sweep_full_{timestamp}.parquet"
    results_df.to_parquet(full_path, index=False)

    # Aggregated results (for heatmap)
    agg_path = output_dir / f"sweep_agg_{timestamp}.parquet"
    agg_df.to_parquet(agg_path, index=False)

    # Also save as CSV for easy inspection
    csv_path = output_dir / f"sweep_agg_{timestamp}.csv"
    agg_df.to_csv(csv_path, index=False)

    # Save config
    config_path = output_dir / f"sweep_config_{timestamp}.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Create symlinks to latest
    for name, path in [
        ("sweep_full_latest.parquet", full_path),
        ("sweep_agg_latest.parquet", agg_path),
        ("sweep_agg_latest.csv", csv_path),
        ("sweep_config_latest.json", config_path),
    ]:
        link = output_dir / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(path.name)

    if verbose:
        print(f"\nSweep complete!")
        print(f"  Full results: {full_path}")
        print(f"  Aggregated: {agg_path}")
        print(f"  CSV: {csv_path}")
        print(f"  Config: {config_path}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run ABM parameter sweep")
    parser.add_argument(
        "--grid-size", type=int, default=6,
        help="Number of points per parameter (default: 6)"
    )
    parser.add_argument(
        "--seeds", type=int, default=2,
        help="Number of seeds per parameter combination (default: 2)"
    )
    parser.add_argument(
        "--households", type=int, default=100,
        help="Number of households per simulation (default: 100)"
    )
    parser.add_argument(
        "--calibration", type=Path, default=None,
        help="Path to calibration.json for calibrated synthetic data"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: outputs/sweeps/uncalibrated or outputs/sweeps/calibrated)"
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
        output_dir = Path("outputs/sweeps/calibrated")
    else:
        output_dir = Path("outputs/sweeps/uncalibrated")

    config = SweepConfig.default_grid(
        grid_size=args.grid_size,
        num_seeds=args.seeds,
        calibration_path=str(args.calibration) if args.calibration else None,
    )
    config.num_households = args.households

    run_sweep(config, output_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()
