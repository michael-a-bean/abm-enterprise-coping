#!/usr/bin/env python3
"""Batch runner for ABM robustness analysis.

Generates multiple simulation runs with different random seeds to assess
stochastic variance. Supports both LSMS-derived and calibrated synthetic data.

Usage:
    # LSMS-derived baseline batch (FLAG 3+4 fix)
    python scripts/run_batch.py --data-source lsms --seeds 10

    # Calibrated synthetic batch (after FLAG 1 fix)
    python scripts/run_batch.py --data-source calibrated --calibration artifacts/calibration/tanzania/calibration.json

    # Quick test
    python scripts/run_batch.py --data-source lsms --seeds 3 --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path


@dataclass
class BatchConfig:
    """Configuration for batch runs."""

    # Data source: lsms | calibrated | synthetic_uncalibrated
    data_source: str = "lsms"

    # For calibrated synthetic
    calibration_path: Path | None = None

    # Run parameters
    country: str = "tanzania"
    scenario: str = "baseline"
    num_waves: int = 4
    seeds: list[int] = field(default_factory=lambda: list(range(1, 11)))

    # For synthetic data only
    num_households: int = 500

    # Output
    output_dir: Path = Path("outputs/batch")

    def validate(self) -> list[str]:
        """Validate configuration, return list of errors."""
        errors = []
        if self.data_source not in ("lsms", "calibrated", "synthetic_uncalibrated"):
            errors.append(f"Invalid data_source: {self.data_source}")
        if self.data_source == "calibrated" and self.calibration_path is None:
            errors.append("calibration_path required for calibrated data_source")
        if self.data_source == "calibrated" and self.calibration_path:
            if not self.calibration_path.exists():
                errors.append(f"Calibration file not found: {self.calibration_path}")
        return errors


@dataclass
class BatchManifest:
    """Aggregate metadata for batch runs."""

    batch_id: str
    timestamp: str
    git_commit: str
    config: dict
    seeds_completed: list[int]
    seeds_failed: list[int]
    total_runs: int
    data_source: str
    run_paths: list[str]


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def run_lsms_simulation(
    country: str,
    scenario: str,
    seed: int,
    num_waves: int,
    output_dir: Path,
    verbose: bool = False,
) -> bool:
    """Run single LSMS-derived simulation."""
    cmd = [
        sys.executable, "-m", "abm_enterprise.cli",
        "run-sim", country,
        "--scenario", scenario,
        "--seed", str(seed),
        "--waves", str(num_waves),
        "--data-dir", "data/processed",
        "--output-dir", str(output_dir),
        "--policy", "none",  # Use RulePolicy or none for baseline
        "--clean-output",
    ]
    if verbose:
        cmd.append("--verbose")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def run_calibrated_simulation(
    calibration_path: Path,
    seed: int,
    num_households: int,
    num_waves: int,
    output_dir: Path,
    verbose: bool = False,
) -> bool:
    """Run single calibrated synthetic simulation."""
    cmd = [
        sys.executable, "-m", "abm_enterprise.cli",
        "run-sim-synthetic", str(calibration_path),
        "--households", str(num_households),
        "--seed", str(seed),
        "--waves", str(num_waves),
        "--output-dir", str(output_dir),
        "--policy", "none",
    ]
    if verbose:
        cmd.append("--verbose")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def run_batch(config: BatchConfig, verbose: bool = True, dry_run: bool = False) -> BatchManifest:
    """Run full batch of simulations."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_id = f"batch_{timestamp}"

    # Create output directory structure
    batch_dir = config.output_dir / config.data_source
    batch_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Batch configuration:")
        print(f"  Data source: {config.data_source}")
        print(f"  Country: {config.country}")
        print(f"  Scenario: {config.scenario}")
        print(f"  Seeds: {config.seeds}")
        print(f"  Output: {batch_dir}")
        if dry_run:
            print("  DRY RUN - no simulations will execute")
        print()

    seeds_completed = []
    seeds_failed = []
    run_paths = []

    for i, seed in enumerate(config.seeds):
        seed_dir = batch_dir / f"seed_{seed}"

        if verbose:
            print(f"[{i+1}/{len(config.seeds)}] Seed {seed}...", end="", flush=True)

        if dry_run:
            print(" [SKIP - dry run]")
            seeds_completed.append(seed)
            run_paths.append(str(seed_dir))
            continue

        # Run simulation based on data source
        if config.data_source == "lsms":
            success = run_lsms_simulation(
                country=config.country,
                scenario=config.scenario,
                seed=seed,
                num_waves=config.num_waves,
                output_dir=seed_dir,
                verbose=False,
            )
        elif config.data_source == "calibrated":
            success = run_calibrated_simulation(
                calibration_path=config.calibration_path,
                seed=seed,
                num_households=config.num_households,
                num_waves=config.num_waves,
                output_dir=seed_dir,
                verbose=False,
            )
        else:
            # synthetic_uncalibrated uses toy mode
            cmd = [
                sys.executable, "-m", "abm_enterprise.cli",
                "run-toy",
                "--seed", str(seed),
                "--output-dir", str(seed_dir),
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                success = result.returncode == 0
            except subprocess.TimeoutExpired:
                success = False

        if success:
            seeds_completed.append(seed)
            run_paths.append(str(seed_dir))
            if verbose:
                print(" OK")
        else:
            seeds_failed.append(seed)
            if verbose:
                print(" FAILED")

    # Create batch manifest
    manifest = BatchManifest(
        batch_id=batch_id,
        timestamp=timestamp,
        git_commit=get_git_commit(),
        config=asdict(config),
        seeds_completed=seeds_completed,
        seeds_failed=seeds_failed,
        total_runs=len(config.seeds),
        data_source=config.data_source,
        run_paths=run_paths,
    )

    # Write batch manifest
    if not dry_run:
        manifest_path = batch_dir / "batch_manifest.json"
        with open(manifest_path, "w") as f:
            # Convert Path to string for JSON serialization
            manifest_dict = asdict(manifest)
            if manifest_dict["config"]["calibration_path"]:
                manifest_dict["config"]["calibration_path"] = str(
                    manifest_dict["config"]["calibration_path"]
                )
            manifest_dict["config"]["output_dir"] = str(manifest_dict["config"]["output_dir"])
            json.dump(manifest_dict, f, indent=2)

        if verbose:
            print(f"\nBatch complete!")
            print(f"  Completed: {len(seeds_completed)}/{len(config.seeds)}")
            print(f"  Failed: {len(seeds_failed)}")
            print(f"  Manifest: {manifest_path}")

    return manifest


def verify_batch_consistency(batch_dir: Path) -> dict:
    """Verify all batch runs have consistent configuration."""
    results = {
        "consistent": True,
        "issues": [],
        "runs_checked": 0,
        "config_summary": {},
    }

    # Find all manifest files
    manifest_files = list(batch_dir.glob("seed_*/tanzania/baseline/manifest.json"))
    if not manifest_files:
        # Try alternate structure
        manifest_files = list(batch_dir.glob("seed_*/manifest.json"))

    if not manifest_files:
        results["issues"].append("No manifest files found")
        results["consistent"] = False
        return results

    # Read first manifest as reference
    with open(manifest_files[0]) as f:
        ref_manifest = json.load(f)

    ref_params = ref_manifest.get("parameters", {})
    results["config_summary"] = {
        "num_households": ref_params.get("num_households"),
        "num_waves": ref_params.get("num_waves"),
        "scenario": ref_manifest.get("scenario"),
        "country": ref_manifest.get("country"),
    }

    # Check all other manifests
    for mf in manifest_files:
        results["runs_checked"] += 1
        with open(mf) as f:
            manifest = json.load(f)
        params = manifest.get("parameters", {})

        # Check consistency
        for key in ["num_households", "num_waves"]:
            if params.get(key) != ref_params.get(key):
                results["issues"].append(
                    f"{mf}: {key} mismatch ({params.get(key)} vs {ref_params.get(key)})"
                )
                results["consistent"] = False

        if manifest.get("scenario") != ref_manifest.get("scenario"):
            results["issues"].append(
                f"{mf}: scenario mismatch ({manifest.get('scenario')} vs {ref_manifest.get('scenario')})"
            )
            results["consistent"] = False

    return results


def main():
    parser = argparse.ArgumentParser(description="Run ABM batch simulations")
    parser.add_argument(
        "--data-source",
        choices=["lsms", "calibrated", "synthetic_uncalibrated"],
        default="lsms",
        help="Data source type (default: lsms)",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Path to calibration.json (required for calibrated data source)",
    )
    parser.add_argument(
        "--country",
        default="tanzania",
        help="Country code (default: tanzania)",
    )
    parser.add_argument(
        "--scenario",
        default="baseline",
        help="Scenario name (default: baseline)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of seeds to run (default: 10)",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=1,
        help="Starting seed number (default: 1)",
    )
    parser.add_argument(
        "--households",
        type=int,
        default=500,
        help="Number of households for synthetic data (default: 500)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/batch"),
        help="Output directory (default: outputs/batch)",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        help="Verify consistency of existing batch directory instead of running",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Verify mode
    if args.verify:
        results = verify_batch_consistency(args.verify)
        print(f"Batch verification: {args.verify}")
        print(f"  Runs checked: {results['runs_checked']}")
        print(f"  Consistent: {results['consistent']}")
        if results["config_summary"]:
            print(f"  Config: {results['config_summary']}")
        if results["issues"]:
            print("  Issues:")
            for issue in results["issues"]:
                print(f"    - {issue}")
        sys.exit(0 if results["consistent"] else 1)

    # Build config
    config = BatchConfig(
        data_source=args.data_source,
        calibration_path=args.calibration,
        country=args.country,
        scenario=args.scenario,
        seeds=list(range(args.start_seed, args.start_seed + args.seeds)),
        num_households=args.households,
        output_dir=args.output_dir,
    )

    # Validate
    errors = config.validate()
    if errors:
        for err in errors:
            print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    # Run batch
    run_batch(config, verbose=not args.quiet, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
