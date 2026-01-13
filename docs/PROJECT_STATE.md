# PROJECT STATE - ABM Enterprise Coping Model

**Last Updated:** 2026-01-12
**Git Commit:** (pending initial commit)
**Current Phase:** Phase 0 - Initialization

---

## Project Overview

Agent-based model (ABM) implementing and validating the mechanisms from "Booms, Busts, and Household Enterprise":
- Cash-crop price busts → enterprise entry as coping mechanism
- Heterogeneity by assets/credit access
- Persistent "stayers" vs intermittent "copers" classification

Validated using LSMS-ISA harmonised dataset, starting with Tanzania and extending to Ethiopia.

---

## Phase Status

| Phase | Name | Status | Acceptance Criteria Met |
|-------|------|--------|------------------------|
| 0 | Initialize Control Docs | COMPLETE | [x] |
| 1 | Scaffold + Tooling + Toy Pipeline | IN PROGRESS | [ ] |
| 2 | LSMS Ingestion + Canonical Panels | NOT STARTED | [ ] |
| 3 | ABM Baseline + Estimand Validation | NOT STARTED | [ ] |
| 4 | LLM Decision Layer | NOT STARTED | [ ] |
| 5 | External Model Reviews | NOT STARTED | [ ] |

---

## Phase 0 TODO Board

- [x] Create repository and initialize git
- [x] Create directory structure
- [x] Create PROJECT_STATE.md
- [x] Create DECISIONS.md
- [x] Create CONOPS.md (Concept of Operations)
- [x] Create VALIDATION_CONTRACT.md
- [x] Create measurement_mapping.csv with placeholder rows
- [x] Initial commit

---

## Phase 1 TODO Board

- [ ] **Agent-Scaffold**: Implement repo layout, Python packaging, CLI
- [ ] **Agent-Scaffold**: Add logging, manifest, Parquet outputs
- [ ] **Agent-Scaffold**: Configure ruff/format, pytest, typing
- [ ] **Agent-Scaffold**: Implement Makefile commands
- [ ] **Agent-Scaffold**: Implement minimal Mesa 3 toy ABM
- [ ] **Agent-R-Scaffold**: Create /analysis Quarto project with renv
- [ ] **Agent-R-Scaffold**: Implement Parquet reader for simulation outputs
- [ ] **Agent-R-Scaffold**: Create simple report confirming schemas
- [ ] Verify: `make test` passes
- [ ] Verify: `make run-toy` generates parquet + manifest
- [ ] Verify: `quarto render` succeeds on toy outputs

---

## Phase 2 TODO Board

- [ ] **Agent-ETL**: Implement LSMS ingestion from GitHub Releases (v2.0)
- [ ] **Agent-ETL**: Create canonical Parquet for 4 levels (plot_crop, plot, household, individual)
- [ ] **Agent-ETL**: Add integrity tests (key uniqueness, referential integrity)
- [ ] **Agent-ETL**: Implement derived targets builder
- [ ] Verify: `make ingest-data` succeeds
- [ ] Verify: `make derive-targets` produces country-specific tables
- [ ] Verify: Tests pass for integrity and schema contracts

---

## Phase 3 TODO Board

- [ ] **Agent-ABM-Core**: Implement household agent model with country configs
- [ ] **Agent-ABM-Core**: Implement RulePolicy coping with feasibility constraints
- [ ] **Agent-ABM-Core**: Implement stayer vs coper classification
- [ ] **Agent-ABM-Core**: Output household-wave outcomes aligned to targets
- [ ] **Agent-Validation-R**: Implement FE regressions
- [ ] **Agent-Validation-R**: Implement distribution comparisons
- [ ] **Agent-Validation-R**: Create portability report (Tanzania vs Ethiopia)
- [ ] Verify: `make run-sim country=tanzania` produces outputs
- [ ] Verify: `make render-report country=tanzania` produces validation report
- [ ] Verify: Ethiopia pipeline works end-to-end

---

## Phase 4 TODO Board

- [ ] **Agent-LLM-Policy**: Implement proposal→constraints→commit interface
- [ ] **Agent-LLM-Policy**: Add prompt/output logging
- [ ] **Agent-LLM-Policy**: Provide stubbed provider adapters
- [ ] **Agent-LLM-Policy**: Ensure deterministic replay option
- [ ] Verify: `make run-sim policy=llm_stub` runs with decision logs
- [ ] Verify: Schema validation rejects malformed actions

---

## Phase 5 TODO Board

- [ ] **Gemini Review**: ODD+D and validation contract coherence
- [ ] **GPT Review**: Reproducibility, logging, schema contracts
- [ ] Incorporate feedback into DECISIONS.md
- [ ] Implement critical fixes

---

## Active Agents

| Agent ID | Name | Task | Status | Branch |
|----------|------|------|--------|--------|
| - | - | - | - | - |

---

## Open Risks / Issues

| ID | Description | Severity | Mitigation | Status |
|----|-------------|----------|------------|--------|
| R1 | LSMS-ISA schema may differ from expected | Medium | Inspect actual files, adapt | Open |
| R2 | Mesa 3 API changes | Low | Pin version, check docs | Open |

---

## Key Decisions Log

See `DECISIONS.md` for detailed decision records.

---

## Commands Reference

```bash
# Setup
make setup              # Install all dependencies

# Quality
make lint               # Run ruff linter
make format             # Format code
make test               # Run pytest

# Data
make ingest-data        # Download and ingest LSMS data
make derive-targets     # Build derived target tables

# Simulation
make run-toy            # Run toy ABM (no real data)
make run-sim            # Run full simulation
make run-sim country=X  # Run for specific country

# Analysis
make render-report      # Generate validation report
```
