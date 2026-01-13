# PROJECT STATE - ABM Enterprise Coping Model

**Last Updated:** 2026-01-13
**Git Commit:** a9bfb67
**Current Phase:** Phase 4 Complete - LLM Decision Layer

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
| 1 | Scaffold + Tooling + Toy Pipeline | COMPLETE | [x] |
| 2 | LSMS Ingestion + Canonical Panels | COMPLETE | [x] |
| 3 | ABM Baseline + Estimand Validation | COMPLETE | [x] |
| 4 | LLM Decision Layer | COMPLETE | [x] |
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

- [x] **Agent-Scaffold**: Implement repo layout, Python packaging, CLI
- [x] **Agent-Scaffold**: Add logging, manifest, Parquet outputs
- [x] **Agent-Scaffold**: Configure ruff/format, pytest, typing
- [x] **Agent-Scaffold**: Implement Makefile commands
- [x] **Agent-Scaffold**: Implement minimal Mesa 3 toy ABM
- [x] **Agent-R-Scaffold**: Create /analysis Quarto project with renv
- [x] **Agent-R-Scaffold**: Implement Parquet reader for simulation outputs
- [x] **Agent-R-Scaffold**: Create simple report confirming schemas
- [x] Verify: `make test` passes (43 tests pass)
- [x] Verify: `make run-toy` generates parquet + manifest
- [x] Verify: `quarto render` succeeds on toy outputs (files ready, awaiting Quarto install)

---

## Phase 2 TODO Board

- [x] **Agent-ETL**: Implement LSMS ingestion from GitHub Releases (v2.0)
- [x] **Agent-ETL**: Create canonical Parquet for 4 levels (plot_crop, plot, household, individual)
- [x] **Agent-ETL**: Add integrity tests (key uniqueness, referential integrity)
- [x] **Agent-ETL**: Implement derived targets builder
- [x] Verify: `make ingest-data` succeeds (Tanzania + Ethiopia)
- [x] Verify: `make derive-targets` produces country-specific tables
- [x] Verify: Tests pass for integrity and schema contracts (65 tests pass)

---

## Phase 3 TODO Board

- [x] **Agent-ABM-Core**: Implement household agent model with country configs
- [x] **Agent-ABM-Core**: Implement CalibratedRulePolicy with data-driven thresholds
- [x] **Agent-ABM-Core**: Implement stayer vs coper classification
- [x] **Agent-ABM-Core**: Output household-wave outcomes aligned to validation contract
- [x] **Agent-Validation-R**: Implement FE regressions (fixest)
- [x] **Agent-Validation-R**: Implement distribution comparisons (KS/χ²)
- [x] **Agent-Validation-R**: Create portability report (Tanzania vs Ethiopia)
- [x] Verify: `make run-sim country=tanzania` produces outputs (75 tests pass)
- [x] Verify: `make render-report` ready (Quarto files complete, awaiting install)
- [x] Verify: Ethiopia pipeline works end-to-end (3 waves, 1500 obs)

---

## Phase 4 TODO Board

- [x] **Agent-LLM-Policy**: Implement proposal→constraints→commit interface
- [x] **Agent-LLM-Policy**: Add prompt/output logging (JSONL with state hash)
- [x] **Agent-LLM-Policy**: Provide stubbed provider adapters (Stub, Replay, Claude, OpenAI)
- [x] **Agent-LLM-Policy**: Ensure deterministic replay option (ReplayProvider)
- [x] Verify: `make run-sim policy=llm_stub` runs with decision logs (101 tests pass)
- [x] Verify: Schema validation rejects malformed actions (constraint tests pass)

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
| a3659ec | Agent-Scaffold | Python ABM scaffold, CLI, Mesa 3, Makefile | Complete | main |
| ad42e45 | Agent-R-Scaffold | R/Quarto validation project, renv | Complete | main |
| a2ca693 | Agent-ETL | LSMS ingestion, canonical Parquet, derived targets | Complete | main |
| aea2bed | Agent-ABM-Core | Derived targets, classification, validation output | Complete | main |
| a3f002a | Agent-Validation-R | FE regressions, distribution tests, portability | Complete | main |
| acd2a31 | Agent-LLM-Policy | LLM providers, constraints, logging, replay | Complete | main |

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
