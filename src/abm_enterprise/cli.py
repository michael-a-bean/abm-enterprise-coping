"""Command-line interface for the ABM Enterprise simulation.

Provides commands for running simulations and validating outputs.
"""

from __future__ import annotations

from pathlib import Path

import typer

from abm_enterprise.data.schemas import PolicyType, SimulationConfig
from abm_enterprise.model import EnterpriseCopingModel
from abm_enterprise.outputs import validate_outputs, write_outputs
from abm_enterprise.utils.logging import get_logger, setup_logging
from abm_enterprise.utils.rng import set_seed

app = typer.Typer(
    name="abm",
    help="ABM Enterprise Coping Simulation CLI",
    add_completion=False,
)

logger = get_logger(__name__)


def _check_output_dir_compatibility(
    output_dir: Path,
    country: str,
    scenario: str,
    num_waves: int,
    clean_output: bool,
) -> None:
    """Check if output directory has compatible configuration.

    If output dir exists with a manifest that has different config,
    warn and fail unless --clean-output is provided.

    Args:
        output_dir: Output directory to check.
        country: Expected country code.
        scenario: Expected scenario name.
        num_waves: Expected number of waves.
        clean_output: If True, delete existing output dir.

    Raises:
        typer.Exit: If config mismatch and clean_output is False.
    """
    import json
    import shutil

    manifest_path = output_dir / "manifest.json"

    # Check if output dir and manifest exist
    if not output_dir.exists() or not manifest_path.exists():
        return  # No existing output, proceed

    # Load existing manifest
    try:
        with open(manifest_path) as f:
            existing_manifest = json.load(f)
    except (json.JSONDecodeError, OSError):
        # Corrupt manifest, treat as stale
        if clean_output:
            shutil.rmtree(output_dir)
            typer.echo(f"Cleaned stale output directory: {output_dir}")
            return
        typer.echo(
            f"ERROR: Output directory {output_dir} has corrupt manifest.\n"
            f"Use --clean-output to remove and regenerate.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Check for config mismatches
    existing_params = existing_manifest.get("parameters", {})
    existing_country = existing_manifest.get("country", "")
    existing_scenario = existing_manifest.get("scenario", "")
    existing_waves = existing_params.get("num_waves", 0)

    mismatches = []
    if existing_country != country:
        mismatches.append(f"country: {existing_country} -> {country}")
    if existing_scenario != scenario:
        mismatches.append(f"scenario: {existing_scenario} -> {scenario}")
    if existing_waves != num_waves:
        mismatches.append(f"num_waves: {existing_waves} -> {num_waves}")

    if mismatches:
        if clean_output:
            shutil.rmtree(output_dir)
            typer.echo(f"Cleaned output directory with mismatched config: {output_dir}")
            typer.echo(f"  Mismatches: {', '.join(mismatches)}")
            return

        typer.echo(
            f"ERROR: Output directory {output_dir} exists with different configuration:\n"
            f"  {chr(10).join('  - ' + m for m in mismatches)}\n"
            f"\n"
            f"This can cause stale parquet partitions from previous runs.\n"
            f"Use --clean-output to remove existing outputs and regenerate,\n"
            f"or specify a different output directory.",
            err=True,
        )
        raise typer.Exit(code=1)


@app.command()
def run_toy(
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    num_households: int = typer.Option(100, "--households", "-n", help="Number of households"),
    num_waves: int = typer.Option(4, "--waves", "-w", help="Number of waves"),
    output_dir: Path = typer.Option(
        Path("outputs/toy"),
        "--output-dir",
        "-o",
        help="Output directory",
    ),
    clean_output: bool = typer.Option(
        False,
        "--clean-output",
        help="Remove existing output directory before writing (prevents stale partitions)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Run simulation with synthetic data.

    Generates synthetic household data and runs the ABM simulation,
    outputting results to the specified directory.

    Example:
        abm run-toy --seed 42 --output-dir outputs/toy
    """
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level, output_dir=output_dir)

    # Check for stale output directory
    _check_output_dir_compatibility(
        output_dir=output_dir,
        country="tanzania",
        scenario="toy",
        num_waves=num_waves,
        clean_output=clean_output,
    )

    typer.echo(f"Running toy simulation with seed={seed}")

    # Initialize RNG
    set_seed(seed)

    # Generate synthetic data
    from abm_enterprise.data.synthetic import generate_synthetic_households

    household_data = generate_synthetic_households(
        n=num_households,
        waves=num_waves,
        country="tanzania",
    )

    # Create config
    config = SimulationConfig(
        country="tanzania",
        scenario="toy",
        seed=seed,
        num_waves=num_waves,
    )

    # Run model
    model = EnterpriseCopingModel(config=config, household_data=household_data)
    model.run()

    # Write outputs
    output_paths = write_outputs(model, output_dir)

    typer.echo(f"Simulation complete. Outputs written to {output_dir}")
    typer.echo(f"  - Outcomes: {output_paths['outcomes']}")
    typer.echo(f"  - Manifest: {output_paths['manifest']}")

    # Summary statistics
    outcomes = model.get_outcomes_dataframe()
    enterprise_rate = outcomes.groupby("wave")["enterprise_status"].mean()
    typer.echo("\nEnterprise participation rate by wave:")
    for wave, rate in enterprise_rate.items():
        typer.echo(f"  Wave {wave}: {rate:.1%}")


@app.command()
def run_sim(
    country: str = typer.Argument(..., help="Country code (tanzania or ethiopia)"),
    scenario: str = typer.Option("baseline", "--scenario", "-S", help="Scenario name"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    num_waves: int = typer.Option(4, "--waves", "-w", help="Number of waves"),
    policy: str = typer.Option(
        "none",
        "--policy",
        "-p",
        help="Policy type (none, credit_access, price_support, asset_transfer, calibrated, llm_stub, llm_replay, llm_claude, llm_openai)",
    ),
    data_dir: Path | None = typer.Option(
        None,
        "--data-dir",
        "-d",
        help="Directory containing processed data (loads derived targets)",
    ),
    output_dir: Path = typer.Option(
        Path("outputs"),
        "--output-dir",
        "-o",
        help="Output directory",
    ),
    calibrate: bool = typer.Option(
        False,
        "--calibrate",
        "-c",
        help="Auto-calibrate policy thresholds from derived data",
    ),
    decision_log_dir: Path | None = typer.Option(
        None,
        "--decision-log-dir",
        help="Directory for LLM decision logs",
    ),
    replay_log: Path | None = typer.Option(
        None,
        "--replay-log",
        help="Path to decision log for replay mode",
    ),
    clean_output: bool = typer.Option(
        False,
        "--clean-output",
        help="Remove existing output directory before writing (prevents stale partitions)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Run simulation with derived targets or synthetic data.

    When --data-dir is provided, loads household_targets.parquet from
    data/processed/{country}/derived/ for validation-aligned simulation.

    Example:
        abm run-sim tanzania --scenario baseline --seed 42
        abm run-sim tanzania --data-dir data/processed --calibrate
        abm run-sim ethiopia --policy credit_access --data-dir data/processed
    """
    log_level = "DEBUG" if verbose else "INFO"
    output_subdir = output_dir / country / scenario
    setup_logging(level=log_level, output_dir=output_subdir)

    typer.echo(f"Running simulation for {country} - {scenario}")

    # Initialize RNG
    set_seed(seed)

    # Load data
    household_data = None
    calibration_thresholds = None

    if data_dir is not None:
        from abm_enterprise.model import (
            compute_calibration_thresholds,
            load_derived_targets,
        )

        try:
            household_data = load_derived_targets(country, data_dir)
            # Infer num_waves from data
            data_waves = sorted(household_data["wave"].unique())
            num_waves = len(data_waves)
            typer.echo(f"Loaded derived targets from {data_dir}")
            typer.echo(f"  Households: {household_data['household_id'].nunique()}")
            typer.echo(f"  Observations: {len(household_data)}")

            # Compute calibration thresholds if requested
            if calibrate:
                calibration_thresholds = compute_calibration_thresholds(household_data)
                typer.echo("Calibrated thresholds:")
                for k, v in calibration_thresholds.items():
                    typer.echo(f"  {k}: {v:.4f}")

        except FileNotFoundError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1) from None
    else:
        from abm_enterprise.data.synthetic import generate_synthetic_households

        household_data = generate_synthetic_households(
            n=100,
            waves=num_waves,
            country=country,
        )
        typer.echo("Using synthetic data (no --data-dir provided)")

    # Check for stale output directory (after num_waves is known)
    _check_output_dir_compatibility(
        output_dir=output_subdir,
        country=country,
        scenario=scenario,
        num_waves=num_waves,
        clean_output=clean_output,
    )

    # Parse policy type and create policy
    from abm_enterprise.policies.llm import LLMPolicyFactory
    from abm_enterprise.policies.rule import CalibratedRulePolicy

    model_policy = None
    policy_type = PolicyType.NONE
    llm_policy_used = False

    # Determine decision log directory for LLM policies
    effective_log_dir = decision_log_dir
    if effective_log_dir is None and policy.startswith("llm_"):
        effective_log_dir = output_dir / country / scenario / "decision_logs"

    if policy == "calibrated" or calibrate:
        # Use calibrated policy
        if calibration_thresholds is not None:
            model_policy = CalibratedRulePolicy(
                country=country,
                thresholds=calibration_thresholds,
            )
            typer.echo("Using CalibratedRulePolicy with data-driven thresholds")
        else:
            model_policy = CalibratedRulePolicy.from_config(country)
            typer.echo("Using CalibratedRulePolicy with config thresholds")

    elif policy == "llm_stub":
        # LLM policy with stub provider
        model_policy = LLMPolicyFactory.create_stub_policy(
            log_dir=effective_log_dir,
            country=country,
        )
        llm_policy_used = True
        typer.echo(f"Using LLMPolicy with StubProvider, logs: {effective_log_dir}")

    elif policy == "llm_replay":
        # LLM policy with replay provider
        if replay_log is None:
            typer.echo("Error: --replay-log required for llm_replay policy", err=True)
            raise typer.Exit(code=1)
        if not replay_log.exists():
            typer.echo(f"Error: Replay log not found: {replay_log}", err=True)
            raise typer.Exit(code=1)
        model_policy = LLMPolicyFactory.create_replay_policy(
            log_path=replay_log,
            log_dir=effective_log_dir,
            country=country,
        )
        llm_policy_used = True
        typer.echo(f"Using LLMPolicy with ReplayProvider from: {replay_log}")

    elif policy == "llm_claude":
        # LLM policy with Claude provider
        import os

        if not os.environ.get("ANTHROPIC_API_KEY"):
            typer.echo("Error: ANTHROPIC_API_KEY environment variable required", err=True)
            raise typer.Exit(code=1)
        model_policy = LLMPolicyFactory.create_claude_policy(
            log_dir=effective_log_dir,
            country=country,
        )
        llm_policy_used = True
        typer.echo(f"Using LLMPolicy with ClaudeProvider, logs: {effective_log_dir}")

    elif policy == "llm_openai":
        # LLM policy with OpenAI provider
        import os

        if not os.environ.get("OPENAI_API_KEY"):
            typer.echo("Error: OPENAI_API_KEY environment variable required", err=True)
            raise typer.Exit(code=1)
        model_policy = LLMPolicyFactory.create_openai_policy(
            log_dir=effective_log_dir,
            country=country,
        )
        llm_policy_used = True
        typer.echo(f"Using LLMPolicy with OpenAIProvider, logs: {effective_log_dir}")

    else:
        try:
            policy_type = PolicyType(policy)
        except ValueError:
            typer.echo(f"Invalid policy type: {policy}", err=True)
            valid_options = [p.value for p in PolicyType] + [
                "calibrated", "llm_stub", "llm_replay", "llm_claude", "llm_openai"
            ]
            typer.echo(f"Valid options: {valid_options}", err=True)
            raise typer.Exit(code=1) from None

    # Create config
    config = SimulationConfig(
        country=country,
        scenario=scenario,
        seed=seed,
        num_waves=num_waves,
        policy_type=policy_type,
    )

    # Run model
    model = EnterpriseCopingModel(
        config=config,
        household_data=household_data,
        policy=model_policy,
    )
    model.run()

    # Write outputs
    output_paths = write_outputs(model, output_subdir)

    typer.echo(f"\nSimulation complete. Outputs written to {output_subdir}")
    typer.echo(f"  - Outcomes: {output_paths['outcomes']}")
    typer.echo(f"  - Manifest: {output_paths['manifest']}")

    # Save LLM decision logs if applicable
    if llm_policy_used and model_policy is not None:
        from abm_enterprise.policies.llm import LLMPolicy

        if isinstance(model_policy, LLMPolicy):
            log_path = model_policy.save_log()
            typer.echo(f"  - Decision log: {log_path}")

            # Print decision summary
            summary = model_policy.get_log_summary()
            typer.echo("\nLLM Decision Summary:")
            typer.echo(f"  Total decisions: {summary['total_decisions']}")
            typer.echo(f"  Constraint failure rate: {summary['constraint_failure_rate']:.1%}")
            for action, count in summary.get('action_counts', {}).items():
                typer.echo(f"  {action}: {count}")

    # Summary statistics for validation contract
    outcomes = model.get_outcomes_dataframe()

    # Enterprise rate by wave
    enterprise_rate = outcomes.groupby("wave")["enterprise_status"].mean()
    typer.echo("\nEnterprise participation rate by wave:")
    for wave, rate in enterprise_rate.items():
        typer.echo(f"  Wave {wave}: {rate:.1%}")

    # Classification distribution
    if "classification" in outcomes.columns:
        class_counts = outcomes.groupby("classification").size()
        typer.echo("\nClassification distribution:")
        for cls, count in class_counts.items():
            pct = count / len(outcomes) * 100
            typer.echo(f"  {cls}: {count} ({pct:.1f}%)")


@app.command()
def validate_schema(
    output_dir: Path = typer.Argument(..., help="Output directory to validate"),
) -> None:
    """Validate output parquet against schemas.

    Checks that:
    - Parquet file exists and is readable
    - Records conform to OutputRecord schema
    - Manifest is valid JSON with required fields

    Example:
        abm validate-schema outputs/toy
    """
    setup_logging(level="INFO")

    typer.echo(f"Validating outputs in {output_dir}")

    results = validate_outputs(output_dir)

    all_valid = True
    for key, value in results.items():
        status = "PASS" if value else "FAIL"
        if not value and "valid" in key:
            all_valid = False
        typer.echo(f"  {key}: {status} ({value})")

    if all_valid:
        typer.echo("\nAll validations passed!")
    else:
        typer.echo("\nSome validations failed!", err=True)
        raise typer.Exit(code=1)


@app.command()
def ingest_data(
    country: str = typer.Option(
        "tanzania",
        "--country",
        "-c",
        help="Country code (tanzania or ethiopia)",
    ),
    output_dir: Path = typer.Option(
        Path("data/processed"),
        "--output-dir",
        "-o",
        help="Output directory for processed data",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-download even if data exists",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Download and process LSMS-ISA data.

    Downloads LSMS-ISA harmonized data from GitHub releases.
    If download fails, generates synthetic data matching the expected schema.

    Example:
        abm ingest-data --country tanzania --output-dir data/processed
        abm ingest-data --country ethiopia --force
    """
    from etl.canonical import create_canonical_tables, validate_canonical_tables
    from etl.ingest import ingest_country_data

    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)

    typer.echo(f"Ingesting LSMS data for {country}")

    try:
        # Step 1: Download/generate raw data
        typer.echo("Step 1: Downloading/generating raw data...")
        raw_path, manifest = ingest_country_data(
            country=country,
            output_dir=output_dir / country,
            force=force,
        )

        if manifest.synthetic_data:
            typer.echo("  Note: Using synthetic data (download failed)")
        else:
            typer.echo("  Downloaded real LSMS data")

        typer.echo(f"  Households: {manifest.num_households}")
        typer.echo(f"  Waves: {manifest.num_waves}")

        # Step 2: Create canonical tables
        typer.echo("\nStep 2: Creating canonical tables...")
        canonical_dir = output_dir / country / "canonical"
        output_paths = create_canonical_tables(
            raw_dir=raw_path,
            output_dir=canonical_dir,
            country=country,
        )

        for table_name, _path in output_paths.items():
            typer.echo(f"  Created: {table_name}.parquet")

        # Step 3: Validate
        typer.echo("\nStep 3: Validating...")
        results = validate_canonical_tables(canonical_dir, country)

        all_valid = all(results.values())
        if all_valid:
            typer.echo("  All validations passed!")
        else:
            for key, value in results.items():
                if not value:
                    typer.echo(f"  FAIL: {key}", err=True)

        typer.echo(f"\nData ingestion complete. Output: {canonical_dir}")

    except Exception as e:
        typer.echo(f"Error during ingestion: {e}", err=True)
        raise typer.Exit(code=1) from None


@app.command()
def derive_targets(
    country: str = typer.Option(
        "tanzania",
        "--country",
        "-c",
        help="Country code (tanzania or ethiopia)",
    ),
    data_dir: Path = typer.Option(
        Path("data/processed"),
        "--data-dir",
        "-d",
        help="Directory containing canonical data",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory (defaults to data_dir/country/derived)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Build derived target tables for ABM validation.

    Creates:
    - enterprise_targets.parquet: Enterprise persistence and classification
    - asset_targets.parquet: Asset index and quintiles
    - price_exposure.parquet: Household price exposure
    - household_targets.parquet: Merged targets for ABM

    Example:
        abm derive-targets --country tanzania --data-dir data/processed
    """
    from etl.derive import build_derived_targets, validate_derived_targets

    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)

    typer.echo(f"Deriving targets for {country}")

    # Set paths
    canonical_dir = data_dir / country / "canonical"
    if output_dir is None:
        output_dir = data_dir / country / "derived"

    if not canonical_dir.exists():
        typer.echo(f"Error: Canonical data not found at {canonical_dir}", err=True)
        typer.echo("Run 'abm ingest-data' first.", err=True)
        raise typer.Exit(code=1)

    try:
        # Build derived tables
        typer.echo("Building derived tables...")
        output_paths = build_derived_targets(
            data_dir=canonical_dir,
            output_dir=output_dir,
            country=country,
        )

        for table_name, _path in output_paths.items():
            typer.echo(f"  Created: {table_name}.parquet")

        # Validate
        typer.echo("\nValidating derived tables...")
        results = validate_derived_targets(output_dir, country)

        all_valid = all(results.values())
        if all_valid:
            typer.echo("  All validations passed!")
        else:
            for key, value in results.items():
                if not value:
                    typer.echo(f"  FAIL: {key}", err=True)

        # Summary statistics
        import polars as pl

        targets_df = pl.read_parquet(output_paths["household_targets"])
        enterprise_df = pl.read_parquet(output_paths["enterprise_targets"])

        typer.echo("\nSummary:")
        typer.echo(f"  Total households: {targets_df['household_id'].n_unique()}")
        typer.echo(f"  Total observations: {len(targets_df)}")

        # Classification breakdown
        classification_counts = (
            enterprise_df
            .group_by("classification")
            .count()
            .sort("classification")
        )
        typer.echo("\nClassification:")
        for row in classification_counts.iter_rows(named=True):
            typer.echo(f"  {row['classification']}: {row['count']}")

        typer.echo(f"\nDerived targets complete. Output: {output_dir}")

    except Exception as e:
        typer.echo(f"Error during derivation: {e}", err=True)
        raise typer.Exit(code=1) from None


@app.command()
def calibrate(
    country: str = typer.Option(
        "tanzania",
        "--country",
        "-c",
        help="Country code (tanzania or ethiopia)",
    ),
    data_dir: Path = typer.Option(
        Path("data/processed"),
        "--data-dir",
        "-d",
        help="Directory containing processed data",
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/calibration"),
        "--output-dir",
        "-o",
        help="Output directory for calibration artifacts",
    ),
    asset_family: str = typer.Option(
        "normal",
        "--asset-family",
        help="Distribution family for assets (normal, lognormal, t)",
    ),
    shock_by_wave: bool = typer.Option(
        False,
        "--shock-by-wave",
        help="Fit separate shock distribution per wave",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Fit calibration distributions from LSMS-derived data.

    Creates calibration.json with fitted distributions for:
    - Asset distribution (normal, lognormal, or t)
    - Price shock distribution (pooled or per-wave)
    - Credit access model (logistic regression)
    - Enterprise baseline statistics
    - Transition rates

    The calibration artifact is used to generate synthetic panels
    that match the distributional characteristics of the real data.

    Example:
        abm calibrate --country tanzania --data-dir data/processed
        abm calibrate --country ethiopia --asset-family t --shock-by-wave
    """
    from abm_enterprise.calibration import fit_calibration

    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)

    typer.echo(f"Calibrating distributions for {country}")

    config = {
        "asset_family": asset_family,
        "shock_by_wave": shock_by_wave,
    }

    try:
        artifact = fit_calibration(
            country=country,
            data_dir=data_dir,
            out_dir=output_dir,
            config=config,
        )

        typer.echo(f"\nCalibration complete!")
        typer.echo(f"  Country: {artifact.country_source}")
        typer.echo(f"  Households: {artifact.n_households}")
        typer.echo(f"  Observations: {artifact.n_observations}")
        typer.echo(f"  Waves: {artifact.waves}")

        typer.echo("\nAssets Distribution:")
        typer.echo(f"  Family: {artifact.assets_distribution.family.value}")
        typer.echo(f"  Mean: {artifact.assets_distribution.params.get('mean', 'N/A'):.4f}")
        typer.echo(f"  Std: {artifact.assets_distribution.params.get('std', 'N/A'):.4f}")

        typer.echo("\nShock Distribution:")
        typer.echo(f"  Pooled Mean: {artifact.shock_distribution.params['mean']:.4f}")
        typer.echo(f"  Pooled Std: {artifact.shock_distribution.params['std']:.4f}")

        typer.echo("\nCredit Model:")
        typer.echo(f"  Intercept: {artifact.credit_model.intercept:.4f}")
        for feat, coef in artifact.credit_model.coefficients.items():
            typer.echo(f"  {feat}: {coef:.4f}")
        typer.echo(f"  Accuracy: {artifact.credit_model.model_metrics.get('accuracy', 'N/A'):.1%}")

        typer.echo("\nEnterprise Baseline:")
        typer.echo(f"  Prevalence: {artifact.enterprise_baseline.prevalence:.1%}")
        typer.echo(f"  Entry Rate: {artifact.enterprise_baseline.entry_rate:.1%}")
        typer.echo(f"  Exit Rate: {artifact.enterprise_baseline.exit_rate:.1%}")

        typer.echo("\nTransition Rates:")
        typer.echo(f"  Enter: {artifact.transition_rates.enter_count} ({artifact.transition_rates.enter_rate:.1%})")
        typer.echo(f"  Exit: {artifact.transition_rates.exit_count} ({artifact.transition_rates.exit_rate:.1%})")
        typer.echo(f"  Stay: {artifact.transition_rates.stay_count} ({artifact.transition_rates.stay_rate:.1%})")

        typer.echo(f"\nArtifact saved to: {output_dir}/{country}/calibration.json")

    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from None
    except Exception as e:
        typer.echo(f"Calibration failed: {e}", err=True)
        raise typer.Exit(code=1) from None


@app.command()
def run_sim_synthetic(
    calibration: Path = typer.Argument(
        ...,
        help="Path to calibration.json artifact",
    ),
    n_households: int = typer.Option(
        1000,
        "--households",
        "-n",
        help="Number of synthetic households",
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    scenario: str = typer.Option("synthetic", "--scenario", "-S", help="Scenario name"),
    policy: str = typer.Option(
        "llm_stub",
        "--policy",
        "-p",
        help="Policy type (llm_stub, llm_o4mini, rule)",
    ),
    llm_temperature: float = typer.Option(
        0.6,
        "--llm-temperature",
        help="Temperature for LLM sampling",
    ),
    llm_k_samples: int = typer.Option(
        5,
        "--llm-k-samples",
        help="Number of samples for LLM voting",
    ),
    cache_decisions: bool = typer.Option(
        True,
        "--cache-decisions/--no-cache",
        help="Enable decision caching",
    ),
    compare_to_lsms: Path | None = typer.Option(
        None,
        "--compare-to-lsms",
        help="Path to LSMS derived targets for comparison",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/synthetic"),
        "--output-dir",
        "-o",
        help="Output directory",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Run synthetic ABM with calibrated distributions and LLM decisions.

    Generates a synthetic panel from the calibration artifact, then runs
    the ABM with the specified policy. Optionally compares outcomes to
    LSMS stylized facts.

    Example:
        abm run-sim-synthetic artifacts/calibration/tanzania/calibration.json \\
          --policy llm_o4mini --households 1000 --llm-k-samples 5

        abm run-sim-synthetic artifacts/calibration/tanzania/calibration.json \\
          --compare-to-lsms data/processed/tanzania/derived/household_targets.parquet
    """
    import json
    import os

    from abm_enterprise.model import run_synthetic_simulation, compare_simulation_to_lsms
    from abm_enterprise.outputs import write_outputs

    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level, output_dir=output_dir)

    typer.echo(f"Running synthetic simulation")
    typer.echo(f"  Calibration: {calibration}")
    typer.echo(f"  Households: {n_households}")
    typer.echo(f"  Policy: {policy}")

    if not calibration.exists():
        typer.echo(f"Error: Calibration file not found: {calibration}", err=True)
        raise typer.Exit(code=1)

    try:
        # Create policy
        if policy == "llm_stub":
            from abm_enterprise.policies.llm import MultiSampleLLMPolicyFactory

            llm_policy = MultiSampleLLMPolicyFactory.create_stub_policy(
                k_samples=llm_k_samples,
                cache_enabled=cache_decisions,
            )
            typer.echo(f"  Using StubProvider (K={llm_k_samples})")

        elif policy in ("llm_o4mini", "llm_openai"):
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                typer.echo("Error: OPENAI_API_KEY required for LLM policy", err=True)
                raise typer.Exit(code=1)

            from abm_enterprise.policies.llm import MultiSampleLLMPolicyFactory

            llm_policy = MultiSampleLLMPolicyFactory.create_o4mini_policy(
                api_key=api_key,
                temperature=llm_temperature,
                k_samples=llm_k_samples,
                cache_enabled=cache_decisions,
            )
            typer.echo(f"  Using o4-mini (T={llm_temperature}, K={llm_k_samples})")

        elif policy == "rule":
            from abm_enterprise.policies.rule import RulePolicy

            llm_policy = RulePolicy()
            typer.echo("  Using RulePolicy (deterministic)")

        else:
            typer.echo(f"Unknown policy: {policy}", err=True)
            raise typer.Exit(code=1)

        # Run synthetic simulation
        model, outcomes = run_synthetic_simulation(
            calibration_path=calibration,
            policy=llm_policy,
            n_households=n_households,
            seed=seed,
            scenario=scenario,
        )

        # Write outputs
        output_subdir = output_dir / scenario
        output_paths = write_outputs(model, output_subdir)

        typer.echo(f"\nSimulation complete. Outputs written to {output_subdir}")
        typer.echo(f"  - Outcomes: {output_paths['outcomes']}")
        typer.echo(f"  - Manifest: {output_paths['manifest']}")

        # Summary statistics
        typer.echo("\nEnterprise participation rate by wave:")
        enterprise_rate = outcomes.groupby("wave")["enterprise_status"].mean()
        for wave, rate in enterprise_rate.items():
            typer.echo(f"  Wave {wave}: {rate:.1%}")

        # Classification distribution
        if "classification" in outcomes.columns:
            class_counts = outcomes.groupby("classification").size()
            typer.echo("\nClassification distribution:")
            for cls, count in class_counts.items():
                pct = count / len(outcomes) * 100
                typer.echo(f"  {cls}: {count} ({pct:.1f}%)")

        # Compare to LSMS if requested
        if compare_to_lsms is not None:
            if not compare_to_lsms.exists():
                typer.echo(f"Warning: LSMS file not found: {compare_to_lsms}", err=True)
            else:
                import pandas as pd

                typer.echo("\nComparing to LSMS stylized facts...")
                lsms_df = pd.read_parquet(compare_to_lsms)

                comparisons = compare_simulation_to_lsms(outcomes, lsms_df)

                # Save comparison
                comparison_path = output_subdir / "lsms_comparison.json"
                with open(comparison_path, "w") as f:
                    json.dump(comparisons, f, indent=2, default=str)
                typer.echo(f"  Comparison saved: {comparison_path}")

                # Print summary
                typer.echo("\n  Enterprise prevalence:")
                sim_prev = comparisons["enterprise_prevalence"]["simulated"]
                obs_prev = comparisons["enterprise_prevalence"]["observed"]
                for wave in sorted(set(sim_prev.keys()) | set(obs_prev.keys())):
                    s = sim_prev.get(wave, 0)
                    o = obs_prev.get(wave, 0)
                    typer.echo(f"    Wave {wave}: Sim={s:.1%} vs Obs={o:.1%}")

                typer.echo("\n  Transition rates:")
                sim_rates = comparisons["transition_rates"]["simulated"]
                obs_rates = comparisons["transition_rates"]["observed"]
                typer.echo(f"    Entry: Sim={sim_rates['enter_rate']:.1%} vs Obs={obs_rates['enter_rate']:.1%}")
                typer.echo(f"    Exit:  Sim={sim_rates['exit_rate']:.1%} vs Obs={obs_rates['exit_rate']:.1%}")

    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from None
    except Exception as e:
        typer.echo(f"Simulation failed: {e}", err=True)
        import traceback
        if verbose:
            traceback.print_exc()
        raise typer.Exit(code=1) from None


@app.command()
def eval_direct(
    train_country: str = typer.Option(
        "tanzania",
        "--train-country",
        "-t",
        help="Country for training baselines",
    ),
    test_country: str | None = typer.Option(
        None,
        "--test-country",
        "-T",
        help="Country for testing (default: same as train)",
    ),
    data_dir: Path = typer.Option(
        Path("data/processed"),
        "--data-dir",
        "-d",
        help="Directory containing processed data",
    ),
    model: str = typer.Option(
        "all",
        "--model",
        "-m",
        help="Model to evaluate (llm_o4mini, baselines, or all)",
    ),
    baselines: str = typer.Option(
        "logit,rf,gbm",
        "--baselines",
        "-b",
        help="Comma-separated list of baselines (logit, rf, gbm)",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/eval"),
        "--output-dir",
        "-o",
        help="Output directory for evaluation results",
    ),
    llm_temperature: float = typer.Option(
        0.6,
        "--llm-temperature",
        help="Temperature for LLM sampling",
    ),
    llm_k_samples: int = typer.Option(
        5,
        "--llm-k-samples",
        help="Number of samples for LLM voting",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Evaluate direct prediction on LSMS transition data.

    Builds a transition dataset from LSMS panel data and evaluates
    LLM predictions against ML baselines. Supports cross-country
    generalization testing (train on one country, test on another).

    Example:
        abm eval-direct --train-country tanzania --test-country ethiopia
        abm eval-direct --train-country tanzania --model baselines
        abm eval-direct --train-country tanzania --model llm_o4mini --llm-k-samples 7
    """
    import json
    import os

    from abm_enterprise.eval import (
        build_transition_dataset,
        train_all_baselines,
        compute_classification_metrics,
        compute_confusion_matrix,
        compute_subgroup_metrics,
        predict_with_llm,
    )
    from abm_enterprise.eval.baselines import prepare_features
    from abm_enterprise.eval.metrics import (
        compare_models,
        save_metrics,
        save_confusion_matrices,
        compute_asset_subgroup_metrics,
        compute_credit_subgroup_metrics,
    )
    from abm_enterprise.eval.direct_prediction import predict_with_baselines

    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)

    # Set test country to train country if not specified
    if test_country is None:
        test_country = train_country

    typer.echo(f"Direct prediction evaluation")
    typer.echo(f"  Train country: {train_country}")
    typer.echo(f"  Test country: {test_country}")
    typer.echo(f"  Model: {model}")

    # Parse baseline list
    baseline_list = [b.strip() for b in baselines.split(",") if b.strip()]

    # Create output directory structure
    if train_country == test_country:
        eval_name = train_country
    else:
        eval_name = f"{train_country}_to_{test_country}"

    eval_output_dir = output_dir / "direct_prediction" / eval_name
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Build transition datasets
        typer.echo("\nStep 1: Building transition datasets...")
        train_df = build_transition_dataset(train_country, data_dir / train_country / "derived")
        typer.echo(f"  Train: {len(train_df)} transitions")

        if test_country != train_country:
            test_df = build_transition_dataset(test_country, data_dir / test_country / "derived")
            typer.echo(f"  Test: {len(test_df)} transitions")
        else:
            test_df = train_df

        # Step 2: Prepare features and labels
        typer.echo("\nStep 2: Preparing features...")
        feature_cols = ["assets_index", "credit_access", "enterprise_status", "price_exposure"]
        X_train, _ = prepare_features(train_df, feature_cols)
        y_train = train_df["transition"].values
        X_test, _ = prepare_features(test_df, feature_cols)
        y_test = test_df["transition"].values

        typer.echo(f"  Feature columns: {feature_cols}")
        typer.echo(f"  X_train shape: {X_train.shape}")
        typer.echo(f"  X_test shape: {X_test.shape}")

        results_df = test_df.copy()
        trained_baselines = {}
        confusion_matrices = {}

        # Step 3: Train and evaluate baselines
        if model in ("all", "baselines"):
            typer.echo("\nStep 3: Training baselines...")
            trained_baselines = train_all_baselines(X_train, y_train, baseline_list)

            # Predict with baselines
            typer.echo("  Running baseline predictions...")
            results_df = predict_with_baselines(results_df, trained_baselines, feature_cols)

            # Compute baseline metrics
            for name in baseline_list:
                pred_col = f"{name}_transition"
                cm = compute_confusion_matrix(y_test, results_df[pred_col])
                confusion_matrices[name] = cm
                metrics = compute_classification_metrics(y_test, results_df[pred_col])
                typer.echo(f"\n  {name} metrics:")
                typer.echo(f"    Accuracy: {metrics.accuracy:.3f}")
                typer.echo(f"    Balanced Accuracy: {metrics.balanced_accuracy:.3f}")
                typer.echo(f"    Macro F1: {metrics.macro_f1:.3f}")

        # Step 4: LLM predictions (if requested)
        if model in ("all", "llm_o4mini", "llm"):
            typer.echo("\nStep 4: Running LLM predictions...")

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                typer.echo("  Warning: OPENAI_API_KEY not set, using stub policy", err=True)
                from abm_enterprise.policies.llm import MultiSampleLLMPolicyFactory

                llm_policy = MultiSampleLLMPolicyFactory.create_stub_policy(
                    k_samples=llm_k_samples,
                )
            else:
                from abm_enterprise.policies.llm import MultiSampleLLMPolicyFactory

                llm_policy = MultiSampleLLMPolicyFactory.create_o4mini_policy(
                    api_key=api_key,
                    temperature=llm_temperature,
                    k_samples=llm_k_samples,
                    cache_enabled=True,
                )

            results_df = predict_with_llm(results_df, llm_policy, batch_size=50)

            # Compute LLM metrics
            cm = compute_confusion_matrix(y_test, results_df["llm_transition"])
            confusion_matrices["llm"] = cm
            metrics = compute_classification_metrics(y_test, results_df["llm_transition"])
            typer.echo(f"\n  LLM metrics:")
            typer.echo(f"    Accuracy: {metrics.accuracy:.3f}")
            typer.echo(f"    Balanced Accuracy: {metrics.balanced_accuracy:.3f}")
            typer.echo(f"    Macro F1: {metrics.macro_f1:.3f}")
            typer.echo(f"    Cache hit rate: {results_df['llm_cache_hit'].mean():.1%}")

        # Step 5: Save results
        typer.echo("\nStep 5: Saving results...")

        # Save predictions
        pred_path = eval_output_dir / "predictions.parquet"
        results_df.to_parquet(pred_path, index=False)
        typer.echo(f"  Predictions: {pred_path}")

        # Save confusion matrices
        save_confusion_matrices(confusion_matrices, eval_output_dir)

        # Compare all models
        pred_cols = []
        if model in ("all", "baselines"):
            pred_cols.extend([f"{b}_transition" for b in baseline_list])
        if model in ("all", "llm_o4mini", "llm"):
            pred_cols.append("llm_transition")

        if pred_cols:
            comparison = compare_models(results_df, "transition", pred_cols)
            comparison_path = eval_output_dir / "model_comparison.csv"
            comparison.to_csv(comparison_path, index=False)
            typer.echo(f"  Model comparison: {comparison_path}")

            typer.echo("\nModel Comparison:")
            typer.echo(comparison.to_string(index=False))

        # Subgroup analysis
        if pred_cols:
            typer.echo("\nStep 6: Subgroup analysis...")

            # By asset quantile
            for pred_col in pred_cols:
                model_name = pred_col.replace("_transition", "")
                asset_metrics = compute_asset_subgroup_metrics(
                    results_df, "transition", pred_col, n_quantiles=4
                )
                asset_path = eval_output_dir / f"subgroup_assets_{model_name}.csv"
                asset_metrics.to_csv(asset_path, index=False)

            # By credit access
            for pred_col in pred_cols:
                model_name = pred_col.replace("_transition", "")
                credit_metrics = compute_credit_subgroup_metrics(
                    results_df, "transition", pred_col
                )
                credit_path = eval_output_dir / f"subgroup_credit_{model_name}.csv"
                credit_metrics.to_csv(credit_path, index=False)

            typer.echo(f"  Subgroup analysis saved to {eval_output_dir}")

        # Save overall metrics
        all_metrics = {}
        for pred_col in pred_cols:
            model_name = pred_col.replace("_transition", "")
            metrics = compute_classification_metrics(y_test, results_df[pred_col])
            all_metrics[model_name] = metrics.to_dict()

        metrics_path = eval_output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        typer.echo(f"  Metrics: {metrics_path}")

        typer.echo(f"\nEvaluation complete. Results in {eval_output_dir}")

    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from None
    except Exception as e:
        typer.echo(f"Evaluation failed: {e}", err=True)
        import traceback
        if verbose:
            traceback.print_exc()
        raise typer.Exit(code=1) from None


@app.command()
def info() -> None:
    """Show information about the ABM package."""
    import mesa

    from abm_enterprise import __version__

    typer.echo("ABM Enterprise Coping Simulation")
    typer.echo(f"  Version: {__version__}")
    typer.echo(f"  Mesa version: {mesa.__version__}")
    typer.echo("\nAvailable commands:")
    typer.echo("  run-toy            Run with synthetic data (quick test)")
    typer.echo("  run-sim            Run with real/synthetic data")
    typer.echo("  run-sim-synthetic  Run synthetic ABM with LLM policy")
    typer.echo("  calibrate          Fit calibration distributions")
    typer.echo("  eval-direct        Evaluate direct prediction")
    typer.echo("  ingest-data        Download and process LSMS data")
    typer.echo("  derive-targets     Build derived target tables")
    typer.echo("  validate-schema    Validate outputs")
    typer.echo("  info               Show this information")


if __name__ == "__main__":
    app()
