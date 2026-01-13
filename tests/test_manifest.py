"""Tests for manifest generation."""

import json
import tempfile
from pathlib import Path

from abm_enterprise.utils.manifest import (
    Manifest,
    generate_manifest,
    get_git_hash,
    load_manifest,
    save_manifest,
)


class TestManifest:
    """Tests for Manifest dataclass."""

    def test_manifest_creation(self) -> None:
        """Test creating a Manifest instance."""
        manifest = Manifest(
            run_id="test_001",
            git_hash="abc123",
            seed=42,
            timestamp="2024-01-01T00:00:00+00:00",
            parameters={"num_waves": 4},
            country="tanzania",
            scenario="baseline",
        )
        assert manifest.run_id == "test_001"
        assert manifest.seed == 42
        assert manifest.parameters["num_waves"] == 4


class TestGenerateManifest:
    """Tests for generate_manifest function."""

    def test_generate_manifest_basic(self) -> None:
        """Test basic manifest generation."""
        manifest = generate_manifest(
            run_id="run_001",
            seed=42,
            country="tanzania",
            scenario="baseline",
        )

        assert manifest.run_id == "run_001"
        assert manifest.seed == 42
        assert manifest.country == "tanzania"
        assert manifest.scenario == "baseline"
        assert manifest.timestamp is not None

    def test_generate_manifest_with_parameters(self) -> None:
        """Test manifest with custom parameters."""
        params = {"num_waves": 4, "policy_type": "credit_access"}
        manifest = generate_manifest(
            run_id="run_002",
            seed=123,
            country="ethiopia",
            scenario="intervention",
            parameters=params,
        )

        assert manifest.parameters == params
        assert manifest.parameters["num_waves"] == 4

    def test_generate_manifest_has_git_hash(self) -> None:
        """Test that manifest includes git hash."""
        manifest = generate_manifest(
            run_id="run_003",
            seed=42,
            country="tanzania",
            scenario="baseline",
        )

        # Git hash should be non-empty (either actual hash or 'unknown')
        assert manifest.git_hash is not None
        assert len(manifest.git_hash) > 0

    def test_generate_manifest_has_version(self) -> None:
        """Test that manifest includes version."""
        manifest = generate_manifest(
            run_id="run_004",
            seed=42,
            country="tanzania",
            scenario="baseline",
        )

        assert manifest.version == "0.1.0"


class TestSaveAndLoadManifest:
    """Tests for save and load manifest functions."""

    def test_save_manifest(self) -> None:
        """Test saving manifest to file."""
        manifest = generate_manifest(
            run_id="save_test",
            seed=42,
            country="tanzania",
            scenario="baseline",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            save_manifest(path, manifest)

            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert data["run_id"] == "save_test"

    def test_load_manifest(self) -> None:
        """Test loading manifest from file."""
        manifest = generate_manifest(
            run_id="load_test",
            seed=123,
            country="ethiopia",
            scenario="test",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            save_manifest(path, manifest)

            loaded = load_manifest(path)
            assert loaded.run_id == "load_test"
            assert loaded.seed == 123
            assert loaded.country == "ethiopia"

    def test_save_creates_directories(self) -> None:
        """Test that save creates parent directories."""
        manifest = generate_manifest(
            run_id="dir_test",
            seed=42,
            country="tanzania",
            scenario="baseline",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "manifest.json"
            save_manifest(path, manifest)

            assert path.exists()

    def test_round_trip(self) -> None:
        """Test save and load round trip."""
        original = generate_manifest(
            run_id="roundtrip",
            seed=999,
            country="tanzania",
            scenario="test",
            parameters={"key": "value", "num": 42},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            save_manifest(path, original)
            loaded = load_manifest(path)

            assert loaded.run_id == original.run_id
            assert loaded.seed == original.seed
            assert loaded.country == original.country
            assert loaded.parameters == original.parameters


class TestGetGitHash:
    """Tests for get_git_hash function."""

    def test_get_git_hash_returns_string(self) -> None:
        """Test that get_git_hash returns a string."""
        git_hash = get_git_hash()
        assert isinstance(git_hash, str)

    def test_get_git_hash_format(self) -> None:
        """Test git hash format (short hash or 'unknown')."""
        git_hash = get_git_hash()
        # Either 'unknown' or a short hex hash
        if git_hash != "unknown":
            # Should be alphanumeric short hash
            assert len(git_hash) >= 7
            assert all(c in "0123456789abcdef" for c in git_hash)
