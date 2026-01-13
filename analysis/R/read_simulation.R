#' Read Simulation Outputs
#'
#' Functions for reading and validating ABM simulation outputs from Parquet format.
#'
#' @author ABM Enterprise Coping Model

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
