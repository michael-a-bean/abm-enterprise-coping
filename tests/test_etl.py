"""Tests for ETL pipeline.

Tests data ingestion, canonical table creation, and derived targets.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import pytest


class TestSchemas:
    """Test ETL schema definitions."""

    def test_country_enum(self):
        """Test Country enum values."""
        from etl.schemas import Country

        assert Country.TANZANIA.value == "tanzania"
        assert Country.ETHIOPIA.value == "ethiopia"
        assert Country("tanzania") == Country.TANZANIA

    def test_country_config(self):
        """Test country configuration."""
        from etl.schemas import COUNTRY_CONFIG, Country

        # Tanzania config
        tz_config = COUNTRY_CONFIG[Country.TANZANIA]
        assert tz_config["waves"] == [1, 2, 3, 4]
        assert len(tz_config["wave_years"]) == 4

        # Ethiopia config
        et_config = COUNTRY_CONFIG[Country.ETHIOPIA]
        assert et_config["waves"] == [1, 2, 3]
        assert len(et_config["wave_years"]) == 3

    def test_household_record_validation(self):
        """Test HouseholdRecord schema validation."""
        from pydantic import ValidationError

        from etl.schemas import HouseholdRecord

        # Valid record
        record = HouseholdRecord(
            household_id="HH001",
            wave=1,
            enterprise_status=1,
            credit_access=0,
            welfare_proxy=1000.0,
            region="Dar es Salaam",
            urban=1,
            household_size=4,
        )
        assert record.household_id == "HH001"

        # Invalid wave
        with pytest.raises(ValidationError):
            HouseholdRecord(
                household_id="HH001",
                wave=-1,  # Invalid
                enterprise_status=1,
                credit_access=0,
                welfare_proxy=1000.0,
                region="Dar es Salaam",
                urban=1,
                household_size=4,
            )

        # Invalid enterprise_status
        with pytest.raises(ValidationError):
            HouseholdRecord(
                household_id="HH001",
                wave=1,
                enterprise_status=2,  # Invalid (should be 0 or 1)
                credit_access=0,
                welfare_proxy=1000.0,
                region="Dar es Salaam",
                urban=1,
                household_size=4,
            )

    def test_plot_record_area_validation(self):
        """Test PlotRecord area validation."""
        from pydantic import ValidationError

        from etl.schemas import PlotRecord

        # Valid record
        record = PlotRecord(
            household_id="HH001",
            wave=1,
            plot_id="P001",
            area_hectares=1.5,
            irrigated=0,
        )
        assert record.area_hectares == 1.5

        # Negative area should fail
        with pytest.raises(ValidationError):
            PlotRecord(
                household_id="HH001",
                wave=1,
                plot_id="P001",
                area_hectares=-1.0,
                irrigated=0,
            )


class TestPrices:
    """Test price data loading and processing."""

    def test_load_crop_prices_tanzania(self):
        """Test loading Tanzania crop prices."""
        from etl.prices import load_crop_prices

        df = load_crop_prices("tanzania")

        assert "crop_code" in df.columns
        assert "crop_name" in df.columns
        assert "year" in df.columns
        assert "price_per_kg" in df.columns

        # Check expected years
        years = df["year"].unique().sort().to_list()
        assert 2008 in years
        assert 2014 in years

    def test_load_crop_prices_ethiopia(self):
        """Test loading Ethiopia crop prices."""
        from etl.prices import load_crop_prices

        df = load_crop_prices("ethiopia")

        years = df["year"].unique().sort().to_list()
        assert 2011 in years
        assert 2015 in years

    def test_compute_price_changes(self):
        """Test price change computation."""
        from etl.prices import compute_price_changes, load_crop_prices

        prices_df = load_crop_prices("tanzania")
        changes_df = compute_price_changes(prices_df, "tanzania")

        assert "crop_code" in changes_df.columns
        assert "wave" in changes_df.columns
        assert "price_change" in changes_df.columns

        # First wave should not have price changes (no previous)
        assert 1 not in changes_df["wave"].unique().to_list()


class TestIngest:
    """Test data ingestion."""

    def test_ingest_generates_synthetic_data(self):
        """Test that ingest generates synthetic data when download fails."""
        from etl.ingest import ingest_country_data

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            raw_path, manifest = ingest_country_data(
                country="tanzania",
                output_dir=output_dir,
                force=True,
            )

            # Should have created synthetic data
            assert manifest.synthetic_data is True
            assert raw_path.exists()

            # Check files exist
            assert (raw_path / "household.parquet").exists()
            assert (raw_path / "individual.parquet").exists()
            assert (raw_path / "plot.parquet").exists()
            assert (raw_path / "plot_crop.parquet").exists()

    def test_synthetic_household_data_structure(self):
        """Test synthetic household data has correct structure."""
        from etl.ingest import ingest_country_data

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            raw_path, _ = ingest_country_data(
                country="tanzania",
                output_dir=output_dir,
                force=True,
            )

            df = pl.read_parquet(raw_path / "household.parquet")

            # Check required columns
            required_cols = [
                "household_id", "wave", "enterprise_status", "credit_access",
                "welfare_proxy", "region", "urban", "household_size"
            ]
            for col in required_cols:
                assert col in df.columns, f"Missing column: {col}"

            # Check data types and constraints
            assert df["enterprise_status"].is_in([0, 1]).all()
            assert df["credit_access"].is_in([0, 1]).all()
            assert df["urban"].is_in([0, 1]).all()
            assert (df["household_size"] >= 1).all()


class TestCanonical:
    """Test canonical table creation and validation."""

    @pytest.fixture
    def setup_test_data(self):
        """Set up test data for canonical tests."""
        from etl.ingest import ingest_country_data

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            raw_path, _ = ingest_country_data(
                country="tanzania",
                output_dir=output_dir,
                force=True,
            )

            yield raw_path, output_dir

    def test_create_canonical_tables(self, setup_test_data):
        """Test canonical table creation."""
        from etl.canonical import create_canonical_tables

        raw_path, output_dir = setup_test_data
        canonical_dir = output_dir / "canonical"

        output_paths = create_canonical_tables(
            raw_dir=raw_path,
            output_dir=canonical_dir,
            country="tanzania",
        )

        assert "household" in output_paths
        assert "individual" in output_paths
        assert "plot" in output_paths
        assert "plot_crop" in output_paths

        # Check files exist
        for path in output_paths.values():
            assert path.exists()

    def test_household_key_uniqueness(self, setup_test_data):
        """Test that (household_id, wave) is unique in household table."""
        from etl.canonical import create_canonical_tables

        raw_path, output_dir = setup_test_data
        canonical_dir = output_dir / "canonical"

        create_canonical_tables(
            raw_dir=raw_path,
            output_dir=canonical_dir,
            country="tanzania",
        )

        df = pl.read_parquet(canonical_dir / "household.parquet")

        # Check uniqueness
        key_counts = df.group_by(["household_id", "wave"]).len()
        duplicates = key_counts.filter(pl.col("len") > 1)

        assert duplicates.height == 0, f"Found {duplicates.height} duplicate keys"

    def test_plot_referential_integrity(self, setup_test_data):
        """Test that all plot.household_id exist in household."""
        from etl.canonical import create_canonical_tables

        raw_path, output_dir = setup_test_data
        canonical_dir = output_dir / "canonical"

        create_canonical_tables(
            raw_dir=raw_path,
            output_dir=canonical_dir,
            country="tanzania",
        )

        household_df = pl.read_parquet(canonical_dir / "household.parquet")
        plot_df = pl.read_parquet(canonical_dir / "plot.parquet")

        # Get household-wave keys
        hh_keys = set(
            household_df
            .select(["household_id", "wave"])
            .unique()
            .iter_rows()
        )

        plot_hh_keys = set(
            plot_df
            .select(["household_id", "wave"])
            .unique()
            .iter_rows()
        )

        # All plot keys should exist in household
        missing = plot_hh_keys - hh_keys
        assert len(missing) == 0, f"Found {len(missing)} plot records with missing households"

    def test_no_negative_areas(self, setup_test_data):
        """Test that plot areas are non-negative."""
        from etl.canonical import create_canonical_tables

        raw_path, output_dir = setup_test_data
        canonical_dir = output_dir / "canonical"

        create_canonical_tables(
            raw_dir=raw_path,
            output_dir=canonical_dir,
            country="tanzania",
        )

        plot_df = pl.read_parquet(canonical_dir / "plot.parquet")
        plot_crop_df = pl.read_parquet(canonical_dir / "plot_crop.parquet")

        assert (plot_df["area_hectares"] >= 0).all()
        assert (plot_crop_df["area_planted"] >= 0).all()

    def test_valid_wave_numbers(self, setup_test_data):
        """Test that wave numbers are valid for country."""
        from etl.canonical import create_canonical_tables
        from etl.schemas import COUNTRY_CONFIG, Country

        raw_path, output_dir = setup_test_data
        canonical_dir = output_dir / "canonical"

        create_canonical_tables(
            raw_dir=raw_path,
            output_dir=canonical_dir,
            country="tanzania",
        )

        valid_waves = set(COUNTRY_CONFIG[Country.TANZANIA]["waves"])

        household_df = pl.read_parquet(canonical_dir / "household.parquet")
        actual_waves = set(household_df["wave"].unique().to_list())

        assert actual_waves.issubset(valid_waves)

    def test_validate_canonical_tables(self, setup_test_data):
        """Test the validation function."""
        from etl.canonical import create_canonical_tables, validate_canonical_tables

        raw_path, output_dir = setup_test_data
        canonical_dir = output_dir / "canonical"

        create_canonical_tables(
            raw_dir=raw_path,
            output_dir=canonical_dir,
            country="tanzania",
        )

        results = validate_canonical_tables(canonical_dir, "tanzania")

        # All validations should pass
        for key, value in results.items():
            assert value is True, f"Validation failed: {key}"


class TestDerive:
    """Test derived target table creation."""

    @pytest.fixture
    def setup_canonical_data(self):
        """Set up canonical data for derive tests."""
        from etl.canonical import create_canonical_tables
        from etl.ingest import ingest_country_data

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            raw_path, _ = ingest_country_data(
                country="tanzania",
                output_dir=output_dir,
                force=True,
            )

            canonical_dir = output_dir / "canonical"
            create_canonical_tables(
                raw_dir=raw_path,
                output_dir=canonical_dir,
                country="tanzania",
            )

            yield canonical_dir, output_dir

    def test_derive_enterprise_targets(self, setup_canonical_data):
        """Test enterprise target derivation."""
        from etl.derive import derive_enterprise_targets

        canonical_dir, _ = setup_canonical_data
        household_df = pl.read_parquet(canonical_dir / "household.parquet")

        targets = derive_enterprise_targets(household_df)

        assert "household_id" in targets.columns
        assert "enterprise_indicator" in targets.columns
        assert "enterprise_persistence" in targets.columns
        assert "classification" in targets.columns

        # Check persistence range
        assert (targets["enterprise_persistence"] >= 0).all()
        assert (targets["enterprise_persistence"] <= 1).all()

        # Check valid classifications
        assert targets["classification"].is_in(["stayer", "coper", "none"]).all()

    def test_derive_asset_targets(self, setup_canonical_data):
        """Test asset target derivation."""
        from etl.derive import derive_asset_targets

        canonical_dir, _ = setup_canonical_data
        household_df = pl.read_parquet(canonical_dir / "household.parquet")

        targets = derive_asset_targets(household_df)

        assert "household_id" in targets.columns
        assert "wave" in targets.columns
        assert "asset_index" in targets.columns
        assert "asset_quintile" in targets.columns

        # Check quintile range
        assert (targets["asset_quintile"] >= 1).all()
        assert (targets["asset_quintile"] <= 5).all()

    def test_build_derived_targets(self, setup_canonical_data):
        """Test full derived target build."""
        from etl.derive import build_derived_targets

        canonical_dir, output_dir = setup_canonical_data
        derived_dir = output_dir / "derived"

        output_paths = build_derived_targets(
            data_dir=canonical_dir,
            output_dir=derived_dir,
            country="tanzania",
        )

        assert "enterprise_targets" in output_paths
        assert "asset_targets" in output_paths
        assert "price_exposure" in output_paths
        assert "household_targets" in output_paths

        # Check household_targets has all required columns
        targets_df = pl.read_parquet(output_paths["household_targets"])
        required_cols = [
            "household_id", "wave", "enterprise_indicator",
            "enterprise_persistence", "classification", "asset_index",
            "asset_quintile", "credit_access", "price_exposure", "welfare_proxy"
        ]
        for col in required_cols:
            assert col in targets_df.columns, f"Missing column: {col}"

    def test_validate_derived_targets(self, setup_canonical_data):
        """Test derived targets validation."""
        from etl.derive import build_derived_targets, validate_derived_targets

        canonical_dir, output_dir = setup_canonical_data
        derived_dir = output_dir / "derived"

        build_derived_targets(
            data_dir=canonical_dir,
            output_dir=derived_dir,
            country="tanzania",
        )

        results = validate_derived_targets(derived_dir, "tanzania")

        # All validations should pass
        for key, value in results.items():
            assert value is True, f"Validation failed: {key}"


class TestIntegration:
    """Integration tests for full ETL pipeline."""

    def test_full_pipeline_tanzania(self):
        """Test complete pipeline for Tanzania."""
        from etl.canonical import create_canonical_tables, validate_canonical_tables
        from etl.derive import build_derived_targets, validate_derived_targets
        from etl.ingest import ingest_country_data

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            country = "tanzania"

            # Step 1: Ingest
            raw_path, manifest = ingest_country_data(
                country=country,
                output_dir=output_dir / "raw",
                force=True,
            )
            assert manifest.num_households > 0

            # Step 2: Canonical
            canonical_dir = output_dir / "canonical"
            create_canonical_tables(
                raw_dir=raw_path,
                output_dir=canonical_dir,
                country=country,
            )
            canonical_results = validate_canonical_tables(canonical_dir, country)
            assert all(canonical_results.values())

            # Step 3: Derive
            derived_dir = output_dir / "derived"
            build_derived_targets(
                data_dir=canonical_dir,
                output_dir=derived_dir,
                country=country,
            )
            derived_results = validate_derived_targets(derived_dir, country)
            assert all(derived_results.values())

    def test_full_pipeline_ethiopia(self):
        """Test complete pipeline for Ethiopia."""
        from etl.canonical import create_canonical_tables, validate_canonical_tables
        from etl.derive import build_derived_targets, validate_derived_targets
        from etl.ingest import ingest_country_data

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            country = "ethiopia"

            # Step 1: Ingest
            raw_path, manifest = ingest_country_data(
                country=country,
                output_dir=output_dir / "raw",
                force=True,
            )
            assert manifest.num_households > 0

            # Step 2: Canonical
            canonical_dir = output_dir / "canonical"
            create_canonical_tables(
                raw_dir=raw_path,
                output_dir=canonical_dir,
                country=country,
            )
            canonical_results = validate_canonical_tables(canonical_dir, country)
            assert all(canonical_results.values())

            # Step 3: Derive
            derived_dir = output_dir / "derived"
            build_derived_targets(
                data_dir=canonical_dir,
                output_dir=derived_dir,
                country=country,
            )
            derived_results = validate_derived_targets(derived_dir, country)
            assert all(derived_results.values())

    def test_output_readable_by_arrow(self):
        """Test that output Parquet files are readable by pyarrow (R compatibility)."""
        import pyarrow.parquet as pq

        from etl.canonical import create_canonical_tables
        from etl.derive import build_derived_targets
        from etl.ingest import ingest_country_data

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Generate data
            raw_path, _ = ingest_country_data(
                country="tanzania",
                output_dir=output_dir / "raw",
                force=True,
            )

            canonical_dir = output_dir / "canonical"
            create_canonical_tables(
                raw_dir=raw_path,
                output_dir=canonical_dir,
                country="tanzania",
            )

            derived_dir = output_dir / "derived"
            build_derived_targets(
                data_dir=canonical_dir,
                output_dir=derived_dir,
                country="tanzania",
            )

            # Read with pyarrow (same as R's arrow package)
            parquet_files = [
                canonical_dir / "household.parquet",
                canonical_dir / "individual.parquet",
                canonical_dir / "plot.parquet",
                canonical_dir / "plot_crop.parquet",
                derived_dir / "household_targets.parquet",
            ]

            for pq_file in parquet_files:
                table = pq.read_table(pq_file)
                assert table.num_rows > 0
                assert table.num_columns > 0
