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
        help="Policy type (none, credit_access, price_support, asset_transfer)",
    ),
    data_dir: Path | None = typer.Option(
        None,
        "--data-dir",
        "-d",
        help="Directory containing real data (uses synthetic if not provided)",
    ),
    output_dir: Path = typer.Option(
        Path("outputs"),
        "--output-dir",
        "-o",
        help="Output directory",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Run simulation with real or synthetic data.

    Example:
        abm run-sim tanzania --scenario baseline --seed 42
        abm run-sim ethiopia --policy credit_access --data-dir data/processed
    """
    log_level = "DEBUG" if verbose else "INFO"
    output_subdir = output_dir / country / scenario
    setup_logging(level=log_level, output_dir=output_subdir)

    typer.echo(f"Running simulation for {country} - {scenario}")

    # Initialize RNG
    set_seed(seed)

    # Parse policy type
    try:
        policy_type = PolicyType(policy)
    except ValueError:
        typer.echo(f"Invalid policy type: {policy}", err=True)
        typer.echo(f"Valid options: {[p.value for p in PolicyType]}", err=True)
        raise typer.Exit(code=1)

    # Load or generate data
    if data_dir is not None:
        from abm_enterprise.model import load_real_data

        try:
            household_data = load_real_data(country, data_dir)
            typer.echo(f"Loaded real data from {data_dir}")
        except FileNotFoundError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
    else:
        from abm_enterprise.data.synthetic import generate_synthetic_households

        household_data = generate_synthetic_households(
            n=100,
            waves=num_waves,
            country=country,
        )
        typer.echo("Using synthetic data (no --data-dir provided)")

    # Create config
    config = SimulationConfig(
        country=country,
        scenario=scenario,
        seed=seed,
        num_waves=num_waves,
        policy_type=policy_type,
    )

    # Run model
    model = EnterpriseCopingModel(config=config, household_data=household_data)
    model.run()

    # Write outputs
    output_paths = write_outputs(model, output_subdir)

    typer.echo(f"\nSimulation complete. Outputs written to {output_subdir}")
    typer.echo(f"  - Outcomes: {output_paths['outcomes']}")
    typer.echo(f"  - Manifest: {output_paths['manifest']}")


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

        for table_name, path in output_paths.items():
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
        raise typer.Exit(code=1)


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

        for table_name, path in output_paths.items():
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
        raise typer.Exit(code=1)


@app.command()
def info() -> None:
    """Show information about the ABM package."""
    import mesa

    from abm_enterprise import __version__

    typer.echo("ABM Enterprise Coping Simulation")
    typer.echo(f"  Version: {__version__}")
    typer.echo(f"  Mesa version: {mesa.__version__}")
    typer.echo("\nAvailable commands:")
    typer.echo("  run-toy          Run with synthetic data")
    typer.echo("  run-sim          Run with real/synthetic data")
    typer.echo("  ingest-data      Download and process LSMS data")
    typer.echo("  derive-targets   Build derived target tables")
    typer.echo("  validate-schema  Validate outputs")
    typer.echo("  info             Show this information")


if __name__ == "__main__":
    app()
