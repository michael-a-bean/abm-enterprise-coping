# Analysis Makefile
# ABM Enterprise Coping Model - R/Quarto Validation

.PHONY: setup render render-country clean help

# Default target
help:
	@echo "ABM Enterprise Coping - Analysis Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  setup          - Restore renv dependencies"
	@echo "  render         - Render validation report (toy mode)"
	@echo "  render-country - Render report for specific country (use country=NAME)"
	@echo "  clean          - Remove rendered outputs"
	@echo ""
	@echo "Examples:"
	@echo "  make setup"
	@echo "  make render"
	@echo "  make render-country country=tanzania"

# Restore renv environment
setup:
	@echo "Restoring renv environment..."
	Rscript -e "renv::restore()"
	@echo "Done."

# Render validation report with default (toy) parameters
render:
	@echo "Rendering validation report..."
	cd quarto && quarto render validation_report.qmd
	@echo "Output: quarto/_output/validation_report.html"

# Render validation report for specific country
# Usage: make render-country country=tanzania
render-country:
ifndef country
	$(error country is not set. Usage: make render-country country=NAME)
endif
	@echo "Rendering validation report for $(country)..."
	cd quarto && quarto render validation_report.qmd \
		-P output_dir:../../outputs/$(country) \
		-P country:$(country)
	@echo "Output: quarto/_output/validation_report.html"

# Clean rendered outputs
clean:
	@echo "Cleaning rendered outputs..."
	rm -rf quarto/_output
	@echo "Done."

# Install renv itself (for fresh setups)
install-renv:
	@echo "Installing renv..."
	Rscript -e "install.packages('renv', repos='https://cloud.r-project.org')"
	@echo "Done."

# Initialize renv (for first-time setup on new machine)
init:
	@echo "Initializing renv..."
	Rscript -e "renv::init(bare = TRUE)"
	Rscript -e "renv::restore()"
	@echo "Done."

# Snapshot current packages (for updating renv.lock)
snapshot:
	@echo "Snapshotting renv..."
	Rscript -e "renv::snapshot()"
	@echo "Updated renv.lock"

# Validate R environment
validate-env:
	@echo "Validating R environment..."
	Rscript -e "library(arrow); library(dplyr); library(ggplot2); library(fixest); library(gt); cat('All packages loaded successfully\n')"

# Run tests
test:
	@echo "Running R tests..."
	Rscript -e "testthat::test_dir('tests')"
