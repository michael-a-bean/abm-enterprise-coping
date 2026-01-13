"""Run manifest generation for tracking simulation provenance.

Manifests capture all metadata needed to reproduce a simulation run,
including git commit hash, seed, parameters, and timestamps.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Manifest:
    """Simulation run manifest for reproducibility.

    Attributes:
        run_id: Unique identifier for this simulation run.
        git_hash: Git commit hash at time of run.
        seed: Random seed used for the simulation.
        timestamp: ISO-formatted timestamp of run start.
        parameters: Dictionary of simulation parameters.
        country: Country code (e.g., 'tanzania', 'ethiopia').
        scenario: Scenario name (e.g., 'baseline', 'policy_intervention').
    """

    run_id: str
    git_hash: str
    seed: int
    timestamp: str
    parameters: dict[str, Any]
    country: str
    scenario: str
    version: str = "0.1.0"
    mesa_version: str = field(default_factory=lambda: _get_mesa_version())


def _get_mesa_version() -> str:
    """Get the installed Mesa version."""
    try:
        import mesa

        return getattr(mesa, "__version__", "unknown")
    except ImportError:
        return "not installed"


def get_git_hash() -> str:
    """Get the current git commit hash.

    Returns:
        The short git commit hash, or 'unknown' if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


def get_git_dirty() -> bool:
    """Check if the git working directory has uncommitted changes.

    Returns:
        True if there are uncommitted changes, False otherwise.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def generate_manifest(
    run_id: str,
    seed: int,
    country: str,
    scenario: str,
    parameters: dict[str, Any] | None = None,
) -> Manifest:
    """Generate a simulation run manifest.

    Args:
        run_id: Unique identifier for this run.
        seed: Random seed for reproducibility.
        country: Country code.
        scenario: Scenario name.
        parameters: Optional dictionary of additional parameters.

    Returns:
        A populated Manifest instance.

    Example:
        >>> manifest = generate_manifest(
        ...     run_id="run_001",
        ...     seed=42,
        ...     country="tanzania",
        ...     scenario="baseline",
        ...     parameters={"num_waves": 4}
        ... )
    """
    git_hash = get_git_hash()
    if get_git_dirty():
        git_hash = f"{git_hash}-dirty"

    timestamp = datetime.now(timezone.utc).isoformat()

    return Manifest(
        run_id=run_id,
        git_hash=git_hash,
        seed=seed,
        timestamp=timestamp,
        parameters=parameters or {},
        country=country,
        scenario=scenario,
    )


def save_manifest(path: Path | str, manifest: Manifest) -> None:
    """Save manifest to JSON file.

    Args:
        path: Path to output JSON file.
        manifest: Manifest instance to save.

    Example:
        >>> manifest = generate_manifest("run_001", 42, "tanzania", "baseline")
        >>> save_manifest(Path("outputs/manifest.json"), manifest)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2, ensure_ascii=False)


def load_manifest(path: Path | str) -> Manifest:
    """Load manifest from JSON file.

    Args:
        path: Path to manifest JSON file.

    Returns:
        A Manifest instance.

    Raises:
        FileNotFoundError: If the manifest file doesn't exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return Manifest(**data)
