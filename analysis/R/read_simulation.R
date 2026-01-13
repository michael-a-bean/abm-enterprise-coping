#' Read Simulation Outputs
#'
#' Functions for reading and validating ABM simulation outputs from Parquet format.
#' Also includes functions for reading derived targets (empirical data).
#'
#' @author ABM Enterprise Coping Model

# Default base directories
DEFAULT_OUTPUT_BASE <- "../../outputs"
DEFAULT_DATA_BASE <- "../../data/processed"

#' Read simulation outputs from Parquet
#'
#' Reads household outcome data and manifest from a simulation output directory.
#'
#' @param output_dir Path to simulation output directory (containing
#'   household_outcomes.parquet and manifest.json)
#' @return List with:
#'   \itemize{
#'     \item outcomes: Data frame of household outcomes
#'     \item manifest: List containing simulation metadata
#'   }
#' @export
#' @examples
#' \dontrun{
#' sim_data <- read_simulation("outputs/toy")
#' }
read_simulation <- function(output_dir) {
  # Validate directory exists
  if (!dir.exists(output_dir)) {
    stop(sprintf("Output directory does not exist: %s", output_dir))
  }

  # Construct file paths
  outcomes_path <- file.path(output_dir, "household_outcomes.parquet")
  manifest_path <- file.path(output_dir, "manifest.json")

  # Validate files exist
  if (!file.exists(outcomes_path)) {
    stop(sprintf("Household outcomes file not found: %s", outcomes_path))
  }
  if (!file.exists(manifest_path)) {
    stop(sprintf("Manifest file not found: %s", manifest_path))
  }

  # Read household outcomes from Parquet
  outcomes <- arrow::read_parquet(outcomes_path)

  # Read manifest JSON
  manifest <- jsonlite::read_json(manifest_path)

  # Validate schema
  validate_schema(outcomes)

  # Return named list
  list(
    outcomes = outcomes,
    manifest = manifest
  )
}

#' Validate simulation output schema
#'
#' Checks that a data frame contains the required columns with correct types
#' for ABM validation analysis.
#'
#' @param df Data frame to validate (typically from household_outcomes.parquet)
#' @return TRUE if valid; throws an error otherwise
#' @export
#' @examples
#' \dontrun{
#' validate_schema(sim_data$outcomes)
#' }
validate_schema <- function(df) {
  # Required columns based on VALIDATION_CONTRACT.md schema
  required_cols <- c(
    "household_id",
    "wave",
    "enterprise_status",
    "price_exposure",
    "assets",
    "credit_access"
  )

  # Check all required columns exist

missing_cols <- setdiff(required_cols, names(df))
  if (length(missing_cols) > 0) {
    stop(sprintf(
      "Missing required columns: %s",
      paste(missing_cols, collapse = ", ")
    ))
  }

  # Validate column types
  type_errors <- character(0)

  # household_id should be character or numeric identifier
  if (!is.character(df$household_id) && !is.numeric(df$household_id)) {
    type_errors <- c(type_errors, "household_id must be character or numeric")
  }

  # wave should be integer/numeric
  if (!is.numeric(df$wave)) {
    type_errors <- c(type_errors, "wave must be numeric")
  }

  # enterprise_status should be logical or 0/1 numeric
  if (!is.logical(df$enterprise_status) && !is.numeric(df$enterprise_status)) {
    type_errors <- c(type_errors, "enterprise_status must be logical or numeric (0/1)")
  }

  # price_exposure should be numeric
  if (!is.numeric(df$price_exposure)) {
    type_errors <- c(type_errors, "price_exposure must be numeric")
  }

  # assets should be numeric
  if (!is.numeric(df$assets)) {
    type_errors <- c(type_errors, "assets must be numeric")
  }

  # credit_access should be logical or 0/1 numeric
  if (!is.logical(df$credit_access) && !is.numeric(df$credit_access)) {
    type_errors <- c(type_errors, "credit_access must be logical or numeric (0/1)")
  }

  # Report any type errors
  if (length(type_errors) > 0) {
    stop(sprintf(
      "Schema validation failed - type errors:\n  %s",
      paste(type_errors, collapse = "\n  ")
    ))
  }

  # Additional validation: check for reasonable values
  if (any(is.na(df$household_id))) {
    warning("Some household_id values are NA")
  }

  if (any(df$wave < 1, na.rm = TRUE)) {
    warning("Some wave values are less than 1")
  }

  TRUE
}

#' List available simulation outputs
#'
#' Scans the outputs directory for valid simulation output directories.
#'
#' @param outputs_base Base directory containing simulation outputs
#'   (default: "outputs")
#' @return Character vector of simulation output directory names
#' @export
list_simulations <- function(outputs_base = "outputs") {
  if (!dir.exists(outputs_base)) {
    return(character(0))
  }

  # List subdirectories
  dirs <- list.dirs(outputs_base, full.names = FALSE, recursive = FALSE)

  # Filter to those with required files
  valid_dirs <- dirs[sapply(dirs, function(d) {
    outcomes_exists <- file.exists(file.path(outputs_base, d, "household_outcomes.parquet"))
    manifest_exists <- file.exists(file.path(outputs_base, d, "manifest.json"))
    outcomes_exists && manifest_exists
  })]

  valid_dirs
}

#' Read derived targets (empirical data)
#'
#' Reads the derived household targets from the processed data directory.
#' These contain empirical data for validation comparison.
#'
#' @param data_dir Base path to processed data (e.g., "data/processed")
#' @param country Country name (e.g., "tanzania", "ethiopia")
#' @return Data frame with derived household targets
#' @export
#' @examples
#' \dontrun{
#' targets <- read_derived_targets("data/processed", "tanzania")
#' }
read_derived_targets <- function(data_dir, country) {
  # Construct path to household targets
  targets_path <- file.path(data_dir, country, "derived", "household_targets.parquet")

  if (!file.exists(targets_path)) {
    stop(sprintf("Derived targets file not found: %s", targets_path))
  }

  # Read parquet file
  targets <- arrow::read_parquet(targets_path)

  # Standardize column names for validation
  # The derived targets use enterprise_indicator, we want enterprise_status for consistency
  if ("enterprise_indicator" %in% names(targets) && !"enterprise_status" %in% names(targets)) {
    targets$enterprise_status <- targets$enterprise_indicator
  }

  # Ensure assets column exists (may be called asset_index in derived)
  if ("asset_index" %in% names(targets) && !"assets" %in% names(targets)) {
    targets$assets <- targets$asset_index
  }

  targets
}

#' Load both simulation and derived targets for validation
#'
#' Convenience function to load both simulation outputs and empirical targets
#' for a given country and scenario.
#'
#' @param output_dir Path to simulation output directory
#' @param data_dir Base path to processed data
#' @param country Country name
#' @return List with simulation (outcomes, manifest) and targets
#' @export
#' @examples
#' \dontrun{
#' data <- load_validation_data(
#'   "outputs/tanzania/baseline",
#'   "data/processed",
#'   "tanzania"
#' )
#' }
load_validation_data <- function(output_dir, data_dir, country) {
  # Read simulation outputs
  sim <- read_simulation(output_dir)

  # Read derived targets
  targets <- tryCatch({
    read_derived_targets(data_dir, country)
  }, error = function(e) {
    warning(sprintf("Could not load derived targets: %s", e$message))
    NULL
  })

  list(
    simulation = sim,
    targets = targets,
    country = country
  )
}

#' Read price exposure data
#'
#' Reads the derived price exposure data for a country.
#'
#' @param data_dir Base path to processed data
#' @param country Country name
#' @return Data frame with price exposure by household and wave
#' @export
read_price_exposure <- function(data_dir, country) {
  price_path <- file.path(data_dir, country, "derived", "price_exposure.parquet")

  if (!file.exists(price_path)) {
    stop(sprintf("Price exposure file not found: %s", price_path))
  }

  arrow::read_parquet(price_path)
}

#' Read enterprise targets
#'
#' Reads the derived enterprise rate targets for a country.
#'
#' @param data_dir Base path to processed data
#' @param country Country name
#' @return Data frame with enterprise rate targets by wave
#' @export
read_enterprise_targets <- function(data_dir, country) {
  ent_path <- file.path(data_dir, country, "derived", "enterprise_targets.parquet")

  if (!file.exists(ent_path)) {
    stop(sprintf("Enterprise targets file not found: %s", ent_path))
  }

  arrow::read_parquet(ent_path)
}

#' Read asset targets
#'
#' Reads the derived asset distribution targets for a country.
#'
#' @param data_dir Base path to processed data
#' @param country Country name
#' @return Data frame with asset distribution targets
#' @export
read_asset_targets <- function(data_dir, country) {
  asset_path <- file.path(data_dir, country, "derived", "asset_targets.parquet")

  if (!file.exists(asset_path)) {
    stop(sprintf("Asset targets file not found: %s", asset_path))
  }

  arrow::read_parquet(asset_path)
}

#' List available countries with derived data
#'
#' Scans the processed data directory for countries with derived targets.
#'
#' @param data_dir Base path to processed data
#' @return Character vector of country names
#' @export
list_countries <- function(data_dir) {
  if (!dir.exists(data_dir)) {
    return(character(0))
  }

  # List subdirectories
  dirs <- list.dirs(data_dir, full.names = FALSE, recursive = FALSE)

  # Filter to those with derived/household_targets.parquet

  valid_dirs <- dirs[sapply(dirs, function(d) {
    targets_exists <- file.exists(
      file.path(data_dir, d, "derived", "household_targets.parquet")
    )
    targets_exists
  })]

  valid_dirs
}

#' Validate derived targets schema
#'
#' Validates that the derived targets data frame has required columns.
#'
#' @param df Data frame to validate
#' @return TRUE if valid; throws error otherwise
#' @export
validate_derived_schema <- function(df) {
  required_cols <- c(
    "household_id",
    "wave",
    "enterprise_indicator",
    "price_exposure"
  )

  missing_cols <- setdiff(required_cols, names(df))
  if (length(missing_cols) > 0) {
    stop(sprintf(
      "Missing required columns in derived targets: %s",
      paste(missing_cols, collapse = ", ")
    ))
  }

  TRUE
}
