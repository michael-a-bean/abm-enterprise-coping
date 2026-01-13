"""Output writing utilities for simulation results.

Handles writing simulation outputs to parquet files and manifests
in a format compatible with R's arrow package.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from abm_enterprise.utils.logging import get_logger
from abm_enterprise.utils.manifest import generate_manifest, save_manifest

if TYPE_CHECKING:
    from abm_enterprise.model import EnterpriseCopingModel

logger = get_logger(__name__)


def write_outputs(
    model: EnterpriseCopingModel,
    output_dir: Path | str,
    partition_by_wave: bool = True,
) -> dict[str, Path]:
    """Write model outputs to files.

    Writes:
    - household_outcomes.parquet (partitioned by wave if specified)
    - manifest.json

    Args:
        model: The completed simulation model.
        output_dir: Directory for output files.
        partition_by_wave: Whether to partition parquet by wave.

    Returns:
        Dictionary mapping output names to file paths.

    Example:
        >>> from abm_enterprise.model import run_toy_simulation
        >>> model, _ = run_toy_simulation(seed=42)
        >>> paths = write_outputs(model, "outputs/toy")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}

    # Get outcomes DataFrame
    outcomes_df = model.get_outcomes_dataframe()

    # Write parquet
    parquet_path = output_dir / "household_outcomes.parquet"
    write_parquet(outcomes_df, parquet_path, partition_by_wave=partition_by_wave)
    outputs["outcomes"] = parquet_path

    logger.info(
        "Wrote outcomes parquet",
        path=str(parquet_path),
        num_records=len(outcomes_df),
    )

    # Generate and write manifest
    manifest = generate_manifest(
        run_id=model.run_id,
        seed=model.config.seed,
        country=model.config.country,
        scenario=model.config.scenario,
        parameters={
            "num_waves": model.config.num_waves,
            "policy_type": model.config.policy_type.value,
            "price_exposure_threshold": model.config.price_exposure_threshold,
            "asset_threshold_percentile": model.config.asset_threshold_percentile,
            "num_households": len(model.agents_by_id),
        },
    )

    manifest_path = output_dir / "manifest.json"
    save_manifest(manifest_path, manifest)
    outputs["manifest"] = manifest_path

    logger.info("Wrote manifest", path=str(manifest_path))

    return outputs


def write_parquet(
    df: pd.DataFrame,
    path: Path | str,
    partition_by_wave: bool = True,
) -> None:
    """Write DataFrame to parquet with R compatibility.

    Uses settings that ensure compatibility with R's arrow package:
    - Uses dictionary encoding for string columns
    - Avoids complex nested types
    - Uses standard compression

    Args:
        df: DataFrame to write.
        path: Output path.
        partition_by_wave: Whether to partition by wave column.
    """
    path = Path(path)

    # Convert to pyarrow table with explicit schema
    table = pa.Table.from_pandas(df, preserve_index=False)

    if partition_by_wave and "wave" in df.columns:
        # Write partitioned dataset
        pq.write_to_dataset(
            table,
            root_path=str(path),
            partition_cols=["wave"],
            compression="snappy",
            use_dictionary=True,
            existing_data_behavior="delete_matching",
        )
    else:
        # Write single file
        pq.write_table(
            table,
            str(path),
            compression="snappy",
            use_dictionary=True,
        )


def read_parquet(path: Path | str) -> pd.DataFrame:
    """Read parquet file or partitioned dataset.

    Args:
        path: Path to parquet file or directory.

    Returns:
        DataFrame with the data.
    """
    path = Path(path)

    if path.is_dir():
        # Read partitioned dataset
        dataset = pq.ParquetDataset(path)
        return dataset.read().to_pandas()
    else:
        # Read single file
        return pq.read_table(path).to_pandas()


def validate_outputs(output_dir: Path | str) -> dict[str, bool]:
    """Validate output files against schemas.

    Args:
        output_dir: Directory containing outputs.

    Returns:
        Dictionary of validation results.
    """
    from abm_enterprise.data.schemas import OutputRecord
    from abm_enterprise.utils.manifest import load_manifest

    output_dir = Path(output_dir)
    results: dict[str, bool] = {}

    # Check parquet file
    parquet_path = output_dir / "household_outcomes.parquet"
    if parquet_path.exists() or (parquet_path.is_dir()):
        try:
            df = read_parquet(parquet_path)
            # Validate each record
            valid = True
            for _, row in df.head(10).iterrows():  # Sample validation
                try:
                    OutputRecord(**row.to_dict())
                except Exception:
                    valid = False
                    break
            results["parquet_valid"] = valid
            results["parquet_num_records"] = len(df)
        except Exception as e:
            logger.error("Parquet validation failed", error=str(e))
            results["parquet_valid"] = False
    else:
        results["parquet_exists"] = False

    # Check manifest
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = load_manifest(manifest_path)
            results["manifest_valid"] = True
            results["manifest_has_git_hash"] = bool(manifest.git_hash)
            results["manifest_run_id"] = manifest.run_id
        except Exception as e:
            logger.error("Manifest validation failed", error=str(e))
            results["manifest_valid"] = False
    else:
        results["manifest_exists"] = False

    return results
