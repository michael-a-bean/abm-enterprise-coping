# RUNBOOK.md - Operational Release Guide

**Last Updated:** 2026-01-13
**Validated On:** runbook/20260113-pipeline-validation branch

---

## Quick Start

```bash
# Full pipeline (toy mode - no external data needed)
make setup
make test
make run-toy
```

---

## Prerequisites

### Python Environment

```bash
# Python 3.11+ required
python --version

# Install in editable mode with dev dependencies
make setup
# OR: pip install -e ".[dev]"
```

### R Environment (for validation reports)

```bash
# R 4.x required
R --version

# Restore renv packages
make setup-r
```

### Quarto (for report rendering)

Quarto is required for rendering validation reports. Version 1.4+ recommended.

```bash
# Check if installed
quarto --version
```

**Installation options:**

1. **Direct download (all platforms):**
   - Visit https://quarto.org/docs/get-started/
   - Download installer for your platform
   - Run installer

2. **Homebrew (macOS):**
   ```bash
   brew install quarto
   ```

3. **apt (Debian/Ubuntu):**
   ```bash
   # Download .deb from quarto.org, then:
   sudo dpkg -i quarto-*.deb
   ```

4. **conda:**
   ```bash
   conda install -c conda-forge quarto
   ```

5. **Docker (no local install needed):**
   ```bash
   # Run report rendering in container
   docker run --rm -v $(pwd):/work -w /work ghcr.io/quarto-dev/quarto \
     quarto render analysis/quarto/validation_report.qmd
   ```

**Verification:**
```bash
quarto --version
# Expected: 1.4.x or higher
```

---

## Pipeline Commands

### 1. Preflight Checks

```bash
# Run test suite (101 tests)
make test

# Expected output:
# ============================= 101 passed in 17.04s =============================

# Run linter
make lint

# Expected output:
# All checks passed!
```

### 2. Toy Simulation

```bash
make run-toy

# Expected outputs:
# - outputs/toy/household_outcomes.parquet (400 rows)
# - outputs/toy/manifest.json
```

### 3. Country Simulation (Tanzania)

```bash
# ETL pipeline (first run only, or when data changes)
make etl-pipeline country=tanzania

# Run simulation
make run-sim COUNTRY=tanzania

# Expected outputs:
# - outputs/tanzania/baseline/household_outcomes.parquet (2000 rows)
# - outputs/tanzania/baseline/manifest.json
```

### 4. Country Simulation (Ethiopia)

```bash
# ETL pipeline
make etl-pipeline country=ethiopia

# Run simulation
make run-sim COUNTRY=ethiopia

# Expected outputs:
# - outputs/ethiopia/baseline/household_outcomes.parquet (1500 rows)
# - outputs/ethiopia/baseline/manifest.json
```

### 5. Validation Reports

```bash
# Toy mode report
make render-report

# Country-specific report
make render-report-country country=tanzania
make render-report-country country=ethiopia

# Expected outputs:
# - analysis/quarto/_output/validation_report.html
```

### 6. LLM Policy Runs

```bash
# Stub policy (deterministic, no API key needed)
make run-sim-llm-stub COUNTRY=tanzania

# Expected outputs:
# - outputs/tanzania/llm_stub/household_outcomes.parquet
# - decision_logs/tanzania/decisions_*.jsonl

# Replay from logs
make run-sim-llm-replay COUNTRY=tanzania REPLAY_LOG=decision_logs/tanzania/decisions_*.jsonl

# Live Claude API (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
make run-sim-llm-claude COUNTRY=tanzania
```

---

## Output Validation

### Manifest Validation

Every run produces a `manifest.json` with required fields:

```json
{
  "run_id": "unique-8-char-id",
  "git_hash": "commit-hash or commit-hash-dirty",
  "seed": 42,
  "timestamp": "ISO-8601 timestamp",
  "country": "tanzania|ethiopia",
  "scenario": "baseline|llm_stub|llm_claude|...",
  "parameters": {...}
}
```

### Parquet Schema

Required columns in `household_outcomes.parquet`:

| Column | Type | Description |
|--------|------|-------------|
| household_id | string | Unique household identifier |
| wave | int | Simulation wave (1-4 for TZ, 1-3 for ET) |
| enterprise_status | bool | Current enterprise operation status |
| enterprise_entry | bool | Entered enterprise this wave |
| price_exposure | float | Weighted crop price change |
| assets | float | Standardized asset index |
| credit_access | int | 0/1 credit access indicator |
| classification | string | stayer/coper/none |

### Quick Schema Check

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('outputs/tanzania/baseline/household_outcomes.parquet')
print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Households: {df.household_id.nunique()}')
"
```

---

## Common Failure Modes

### 1. Quarto Not Found

**Symptom:**
```
==========================================
ERROR: Quarto is not installed or not in PATH
==========================================
```

Or older error format:
```
make: *** [Makefile:167: render-report] Error 127
```

**Solution:**
The Makefile now provides detailed installation instructions when Quarto is missing.
See the "Quarto (for report rendering)" section in Prerequisites above.

Quick fix options:
1. Install Quarto: https://quarto.org/docs/get-started/
2. Use Docker: `docker run --rm -v $(pwd):/work ghcr.io/quarto-dev/quarto quarto render ...`
3. Skip report rendering if only simulation validation is needed

### 2. R Packages Missing

**Symptom:**
```
Error in library(fixest) : there is no package called 'fixest'
```

**Solution:**
```bash
make setup-r
# OR: cd analysis && R -e "renv::restore()"
```

### 3. Stale Parquet Partitions

**Symptom:**
Row counts don't match manifest (e.g., 2000 rows when expecting 1500), OR
error message about mismatched configuration:
```
ERROR: Output directory outputs/ethiopia/baseline exists with different configuration:
  - num_waves: 4 -> 3
```

**Cause:**
Previous run with different wave count left old partitions

**Solution (automatic):**
```bash
# Use --clean-output flag to automatically remove stale outputs
make run-sim COUNTRY=ethiopia CLEAN_OUTPUT=1

# Or via CLI directly
abm run-sim ethiopia --clean-output
```

**Solution (manual):**
```bash
rm -rf outputs/ethiopia/baseline/
make run-sim COUNTRY=ethiopia
```

**Prevention:**
The CLI now automatically detects config mismatches and fails with a clear error.
Use `CLEAN_OUTPUT=1` to enable automatic cleanup.

### 4. LLM API Key Missing

**Symptom:**
```
Error: ANTHROPIC_API_KEY environment variable not set
```

**Solution:**
```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...
# OR use stub policy: make run-sim-llm-stub
```

### 5. Lint Failures

**Symptom:**
```
Found 44 errors.
```

**Solution:**
```bash
ruff check src/ tests/ --fix  # Auto-fix most issues
make lint                      # Verify clean
```

---

## Decision Log Analysis

### Log Location
```
decision_logs/<country>/decisions_YYYYMMDD_HHMMSS.jsonl
```

### Log Fields
Each JSONL record contains:
- `state_hash`: 32-char SHA-256 for state verification
- `prompt`: Full LLM prompt
- `response`: Raw LLM response
- `parsed_action`: Action parsed from response
- `constraints_passed`: Boolean
- `failed_constraints`: List of failing constraint names
- `final_action`: Committed action (may differ if constraints failed)

### Spot-Check Script

```bash
python3 -c "
import json
with open('decision_logs/tanzania/decisions_*.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if not r['constraints_passed'] and r['final_action'] != 'NO_CHANGE':
            print(f'ERROR: Infeasible action committed')
            exit(1)
print('All constraint failures properly handled')
"
```

---

## Release Checklist

### Pre-Release Validation

- [ ] `make test` - 101 tests passing
- [ ] `make lint` - No lint errors
- [ ] Toy run produces valid outputs
- [ ] Tanzania baseline produces 2000 rows (500 HH × 4 waves)
- [ ] Ethiopia baseline produces 1500 rows (500 HH × 3 waves)
- [ ] LLM stub run produces decision logs
- [ ] No infeasible actions committed in decision logs
- [ ] Manifests contain git_hash, seed, country, scenario

### Documentation

- [ ] CLAUDE.md reflects current architecture
- [ ] RUN_LOG.md documents latest validation run
- [ ] CODE_REVIEW.md updated with findings
- [ ] EXTERNAL_REVIEW.md contains model feedback

### Code Quality

- [ ] All critical issues from CODE_REVIEW.md addressed
- [ ] Critical items from EXTERNAL_REVIEW.md implemented
- [ ] State hash uses 32+ hex chars
- [ ] R FE regressions use clustered standard errors

### Reproducibility

- [ ] Seeds explicitly set in all runs
- [ ] Git hash captured in manifest
- [ ] Decision logs enable replay
- [ ] renv.lock captures R dependencies

---

## Troubleshooting Commands

```bash
# Check output directory structure
tree outputs/ -L 3

# Validate specific manifest
cat outputs/tanzania/baseline/manifest.json | python3 -m json.tool

# Count parquet rows
python3 -c "import pandas as pd; print(len(pd.read_parquet('outputs/tanzania/baseline/household_outcomes.parquet')))"

# Check decision log summary
python3 -c "
import json
from collections import Counter
actions = Counter()
with open('decision_logs/tanzania/decisions_*.jsonl') as f:
    for line in f:
        actions[json.loads(line)['final_action']] += 1
print(dict(actions))
"

# Clear all outputs for fresh run
make clean-outputs
```

---

## Contact

For issues, see: https://github.com/anthropics/claude-code/issues
