# DECISIONS LOG - ABM Enterprise Coping Model

This document records key architectural and design decisions, including rationale and any subsequent revisions based on external reviews.

---

## Decision Record Format

Each decision follows this structure:
- **ID**: Unique identifier (DEC-XXX)
- **Date**: When decision was made
- **Context**: What prompted the decision
- **Decision**: What was decided
- **Rationale**: Why this choice
- **Alternatives Considered**: Other options evaluated
- **Consequences**: Expected outcomes
- **Review Status**: Pending/Reviewed by Gemini/GPT
- **Revisions**: Any changes after review

---

## DEC-001: Repository Structure

**Date:** 2026-01-12
**Context:** Need clean architecture supporting ABM + ETL + validation

**Decision:**
```
/
├── src/
│   ├── abm_enterprise/     # Python ABM package
│   │   ├── agents/         # Agent implementations
│   │   ├── policies/       # Decision policies (Rule, LLM)
│   │   ├── data/           # Data loaders and interfaces
│   │   └── utils/          # Logging, manifest, RNG
│   └── etl/                # Data ingestion pipelines
├── analysis/
│   └── quarto/             # R/Quarto validation reports
├── config/                 # Country configs, scenario definitions
├── data/                   # Local data (gitignored)
├── docs/                   # Project documentation
└── tests/                  # Python test suite
```

**Rationale:**
- Separates concerns (ABM core, ETL, analysis)
- Python package structure enables clean imports
- R analysis isolated in dedicated Quarto project
- Config externalized for country extensibility

**Alternatives Considered:**
- Monolithic scripts: Rejected (poor maintainability)
- Separate repos for ABM and analysis: Rejected (coupling needed)

**Consequences:**
- Clear boundaries for sub-agent work
- May need cross-language data contracts

**Review Status:** Pending Gemini review (Phase 1 end)

---

## DEC-002: Technology Stack

**Date:** 2026-01-12
**Context:** Need modern, reproducible stack for research-grade ABM

**Decision:**
- **Python**: Mesa 3.x for ABM, pandas/polars for data
- **R**: Quarto + fixest for validation regressions
- **Package Management**: bun (JS if needed), uv/pip for Python, renv for R
- **Data Format**: Parquet for all intermediate/output data
- **Build**: Makefile for unified commands

**Rationale:**
- Mesa 3 is current standard for Python ABM
- Parquet enables efficient cross-language data sharing
- renv ensures R reproducibility
- Makefile provides single entry point regardless of underlying tool

**Alternatives Considered:**
- NetLogo: Rejected (harder to integrate with data pipelines)
- CSV outputs: Rejected (inefficient, no schema enforcement)

**Consequences:**
- Dual-language environment complexity
- Need Parquet readers in both Python and R

**Review Status:** Pending

---

## DEC-003: Measurement Mapping as Contract

**Date:** 2026-01-12
**Context:** ABM outputs must align precisely with LSMS-derived targets

**Decision:**
- Maintain `docs/measurement_mapping.csv` as authoritative contract
- ABM variable ↔ Dataset variable ↔ Transformation ↔ Units
- Generate schema validators from this mapping
- Validation reports reference this mapping

**Rationale:**
- Explicit mapping prevents drift between simulation and data
- Enables automated schema checking
- Documents all transformations for reproducibility

**Alternatives Considered:**
- Implicit matching by column name: Rejected (too error-prone)
- Separate documentation: Rejected (drift risk)

**Consequences:**
- Must update mapping when adding new variables
- Automated tests can verify compliance

**Review Status:** Pending GPT review

---

## DEC-004: Reproducibility Architecture

**Date:** 2026-01-12
**Context:** Research-grade work requires full reproducibility

**Decision:**
- Centralized RNG seeding with explicit seed in manifest
- run_id includes git commit hash
- All outputs include manifest.json with full provenance
- LLM policy logs all prompts/outputs with hashes
- Deterministic replay mode available

**Rationale:**
- Any run can be exactly reproduced
- Provenance chain from data through analysis
- LLM non-determinism captured for audit

**Alternatives Considered:**
- Per-component random seeds: Rejected (harder to coordinate)
- No LLM logging: Rejected (loses reproducibility)

**Consequences:**
- Slight overhead for logging
- Storage for decision logs

**Review Status:** Pending GPT review

---

## DEC-005: Country Configuration Pattern

**Date:** 2026-01-12
**Context:** Must support Tanzania first, then extend to Ethiopia

**Decision:**
- YAML configuration files per country (tanzania.yaml, ethiopia.yaml)
- Configuration includes: wave mappings, crop codes, variable mappings
- ABM and ETL parameterized by country config
- No country-specific code paths in core logic

**Rationale:**
- Adding Ethiopia should require only new config, not code changes
- Enables future country extensions
- Clear separation of domain knowledge from logic

**Alternatives Considered:**
- Country-specific modules: Rejected (code duplication)
- Single hardcoded country: Rejected (not extensible)

**Consequences:**
- Configuration validation needed
- Initial Tanzania work must be properly parameterized

**Review Status:** Pending

---

## Pending Decisions

- [ ] LLM provider interface design
- [ ] Price series data source and format
- [ ] Exact LSMS release version and variable selection
- [ ] Stayer/coper classification thresholds

---

## Review Feedback Log

### Gemini Reviews

(To be populated after Phase 1, 2, 3, 4 completions)

### GPT Reviews

(To be populated after Phase 1, 2, 3, 4 completions)
