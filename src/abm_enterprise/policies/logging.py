"""Decision logging system for LLM-based policies.

Provides logging of all LLM prompts, responses, and final actions
for reproducibility and analysis.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from abm_enterprise.data.schemas import HouseholdState
from abm_enterprise.policies.base import Action


@dataclass
class DecisionRecord:
    """Record of a single LLM-based decision.

    Captures all information needed to reproduce and analyze
    the decision process.

    Attributes:
        timestamp: ISO format timestamp of the decision.
        household_id: Unique household identifier.
        wave: Simulation wave number.
        state_hash: SHA-256 hash of input state for verification.
        prompt: Full prompt sent to LLM.
        response: Raw response from LLM.
        parsed_action: Action parsed from response.
        constraints_passed: Whether all constraints were satisfied.
        failed_constraints: List of constraint names that failed.
        final_action: Action actually applied (may differ if constraints failed).
        provider: Name of the LLM provider used.
        model: Model name/ID used.
        latency_ms: Response latency in milliseconds.
        metadata: Additional metadata (e.g., token counts).
    """

    timestamp: str
    household_id: str
    wave: int
    state_hash: str
    prompt: str
    response: str
    parsed_action: str
    constraints_passed: bool
    failed_constraints: list[str] = field(default_factory=list)
    final_action: str = ""
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the record.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionRecord:
        """Create record from dictionary.

        Args:
            data: Dictionary with record fields.

        Returns:
            DecisionRecord instance.
        """
        return cls(**data)


def compute_state_hash(state: HouseholdState) -> str:
    """Compute SHA-256 hash of household state.

    Args:
        state: Household state to hash.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    # Create deterministic string representation
    state_str = (
        f"{state.household_id}|{state.wave}|{state.assets:.6f}|"
        f"{state.credit_access}|{state.enterprise_status.value}|"
        f"{state.price_exposure:.6f}"
    )
    return hashlib.sha256(state_str.encode()).hexdigest()[:16]


class DecisionLogger:
    """Logger for LLM decision records.

    Maintains a list of decision records and provides methods
    for saving and loading in JSONL format.

    Attributes:
        output_dir: Directory for saving log files.
        records: List of logged decision records.
        session_id: Unique identifier for this logging session.
    """

    def __init__(
        self,
        output_dir: Path | str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Initialize decision logger.

        Args:
            output_dir: Directory for saving logs. If None, logging is in-memory only.
            session_id: Unique session identifier. Generated if not provided.
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.records: list[DecisionRecord] = []

        if session_id is None:
            self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        else:
            self.session_id = session_id

        # Create output directory if specified
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        state: HouseholdState,
        prompt: str,
        response: str,
        parsed_action: str | None,
        constraints_passed: bool,
        failed_constraints: list[str] | None = None,
        final_action: Action | str | None = None,
        provider: str = "",
        model: str = "",
        latency_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> DecisionRecord:
        """Log a decision.

        Args:
            state: Household state at decision time.
            prompt: Prompt sent to LLM.
            response: Raw LLM response.
            parsed_action: Action parsed from response (or None if parsing failed).
            constraints_passed: Whether all constraints passed.
            failed_constraints: Names of failed constraints.
            final_action: Action actually applied.
            provider: LLM provider name.
            model: Model name/ID.
            latency_ms: Response latency.
            metadata: Additional metadata.

        Returns:
            The created DecisionRecord.
        """
        # Handle action string conversion
        if isinstance(final_action, Action):
            final_action_str = final_action.value
        elif final_action is not None:
            final_action_str = str(final_action)
        else:
            final_action_str = parsed_action or "NO_CHANGE"

        record = DecisionRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            household_id=state.household_id,
            wave=state.wave,
            state_hash=compute_state_hash(state),
            prompt=prompt,
            response=response,
            parsed_action=parsed_action or "",
            constraints_passed=constraints_passed,
            failed_constraints=failed_constraints or [],
            final_action=final_action_str,
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

        self.records.append(record)
        return record

    def save(self, path: Path | str | None = None) -> Path:
        """Save records to JSONL file.

        Args:
            path: Output file path. If None, uses default in output_dir.

        Returns:
            Path to saved file.

        Raises:
            ValueError: If no output path available.
        """
        if path is None:
            if self.output_dir is None:
                raise ValueError("No output path specified and no output_dir set")
            path = self.output_dir / f"decisions_{self.session_id}.jsonl"
        else:
            path = Path(path)

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(record.to_dict()) + "\n")

        return path

    def save_incremental(self, record: DecisionRecord) -> None:
        """Append a single record to the log file.

        Useful for streaming writes during simulation.

        Args:
            record: Record to append.
        """
        if self.output_dir is None:
            return

        path = self.output_dir / f"decisions_{self.session_id}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    @classmethod
    def load(cls, path: Path | str) -> list[DecisionRecord]:
        """Load decision records from JSONL file.

        Args:
            path: Path to JSONL file.

        Returns:
            List of DecisionRecord instances.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Log file not found: {path}")

        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    records.append(DecisionRecord.from_dict(data))

        return records

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics of logged decisions.

        Returns:
            Dictionary with summary statistics.
        """
        if not self.records:
            return {
                "total_decisions": 0,
                "unique_households": 0,
                "waves": [],
                "action_counts": {},
                "constraint_failure_rate": 0.0,
            }

        action_counts: dict[str, int] = {}
        for r in self.records:
            action = r.final_action
            action_counts[action] = action_counts.get(action, 0) + 1

        failed_count = sum(1 for r in self.records if not r.constraints_passed)

        return {
            "total_decisions": len(self.records),
            "unique_households": len({r.household_id for r in self.records}),
            "waves": sorted({r.wave for r in self.records}),
            "action_counts": action_counts,
            "constraint_failure_rate": failed_count / len(self.records),
        }

    def clear(self) -> None:
        """Clear all logged records."""
        self.records.clear()
