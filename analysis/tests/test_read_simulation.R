# Test file for read_simulation.R
# ABM Enterprise Coping Model
#
# Run with: Rscript -e "testthat::test_file('tests/test_read_simulation.R')"
# Or:       make test (from analysis directory)

library(testthat)
library(arrow)
library(dplyr)

# Source the functions we're testing
source("../R/read_simulation.R")
source("../R/validation_helpers.R")

# ============================================================
# Test fixtures
# ============================================================

#' Create a minimal valid test data frame
create_valid_df <- function(n_households = 10, n_waves = 4) {
  expand.grid(
    household_id = sprintf("HH%03d", 1:n_households),
    wave = 1:n_waves
  ) |>
    dplyr::as_tibble() |>
    dplyr::mutate(
      enterprise_status = sample(c(TRUE, FALSE), n(), replace = TRUE),
      price_exposure = rnorm(n(), mean = -0.05, sd = 0.1),
      assets = abs(rnorm(n(), mean = 1000, sd = 500)),
      credit_access = sample(c(TRUE, FALSE), n(), replace = TRUE, prob = c(0.3, 0.7))
    )
}

#' Create a test output directory with parquet and manifest
create_test_output_dir <- function(dir_path, df = NULL, manifest = NULL) {
  # Create directory if needed
  dir.create(dir_path, showWarnings = FALSE, recursive = TRUE)

  # Use default data if not provided
  if (is.null(df)) {
    df <- create_valid_df()
  }

  # Use default manifest if not provided
  if (is.null(manifest)) {
    manifest <- list(
      model_version = "0.1.0",
      country = "test",
      n_households = length(unique(df$household_id)),
      n_waves = length(unique(df$wave)),
      timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
    )
  }

  # Write parquet
  arrow::write_parquet(df, file.path(dir_path, "household_outcomes.parquet"))

  # Write manifest
  jsonlite::write_json(manifest, file.path(dir_path, "manifest.json"), auto_unbox = TRUE)

  invisible(dir_path)
}

# ============================================================
# Tests for validate_schema
# ============================================================

test_that("validate_schema passes for valid data frame", {
  df <- create_valid_df()
  expect_true(validate_schema(df))
})

test_that("validate_schema fails when missing required columns", {
  df <- create_valid_df()

  # Remove household_id
  df_missing_id <- df |> dplyr::select(-household_id)
  expect_error(validate_schema(df_missing_id), "Missing required columns.*household_id")

  # Remove wave
  df_missing_wave <- df |> dplyr::select(-wave)
  expect_error(validate_schema(df_missing_wave), "Missing required columns.*wave")

  # Remove enterprise_status
  df_missing_ent <- df |> dplyr::select(-enterprise_status)
  expect_error(validate_schema(df_missing_ent), "Missing required columns.*enterprise_status")

  # Remove price_exposure
  df_missing_price <- df |> dplyr::select(-price_exposure)
  expect_error(validate_schema(df_missing_price), "Missing required columns.*price_exposure")

  # Remove assets
  df_missing_assets <- df |> dplyr::select(-assets)
  expect_error(validate_schema(df_missing_assets), "Missing required columns.*assets")

  # Remove credit_access
  df_missing_credit <- df |> dplyr::select(-credit_access)
  expect_error(validate_schema(df_missing_credit), "Missing required columns.*credit_access")
})

test_that("validate_schema fails when columns have wrong types", {
  df <- create_valid_df()

  # wave should be numeric
  df_bad_wave <- df |> dplyr::mutate(wave = as.character(wave))
  expect_error(validate_schema(df_bad_wave), "wave must be numeric")

  # price_exposure should be numeric
  df_bad_price <- df |> dplyr::mutate(price_exposure = as.character(price_exposure))
  expect_error(validate_schema(df_bad_price), "price_exposure must be numeric")

  # assets should be numeric
  df_bad_assets <- df |> dplyr::mutate(assets = as.character(assets))
  expect_error(validate_schema(df_bad_assets), "assets must be numeric")
})

test_that("validate_schema accepts numeric 0/1 for boolean columns", {
  df <- create_valid_df() |>
    dplyr::mutate(
      enterprise_status = as.integer(enterprise_status),
      credit_access = as.integer(credit_access)
    )
  expect_true(validate_schema(df))
})

test_that("validate_schema warns on NA household_id", {
  df <- create_valid_df()
  df$household_id[1] <- NA
  expect_warning(validate_schema(df), "Some household_id values are NA")
})

test_that("validate_schema warns on wave values less than 1", {
  df <- create_valid_df()
  df$wave[1] <- 0
  expect_warning(validate_schema(df), "Some wave values are less than 1")
})

# ============================================================
# Tests for read_simulation
# ============================================================

test_that("read_simulation fails for non-existent directory", {
  expect_error(read_simulation("/non/existent/path"), "Output directory does not exist")
})

test_that("read_simulation fails when parquet file missing", {
  test_dir <- tempfile()
  dir.create(test_dir)

  # Only create manifest, not parquet
  jsonlite::write_json(list(test = TRUE), file.path(test_dir, "manifest.json"))

  expect_error(read_simulation(test_dir), "Household outcomes file not found")

  # Cleanup
  unlink(test_dir, recursive = TRUE)
})

test_that("read_simulation fails when manifest file missing", {
  test_dir <- tempfile()
  dir.create(test_dir)

  # Only create parquet, not manifest
  df <- create_valid_df()
  arrow::write_parquet(df, file.path(test_dir, "household_outcomes.parquet"))

  expect_error(read_simulation(test_dir), "Manifest file not found")

  # Cleanup
  unlink(test_dir, recursive = TRUE)
})

test_that("read_simulation works with valid input", {
  test_dir <- tempfile()
  create_test_output_dir(test_dir)

  result <- read_simulation(test_dir)

  expect_type(result, "list")
  expect_true("outcomes" %in% names(result))
  expect_true("manifest" %in% names(result))
  expect_s3_class(result$outcomes, "data.frame")
  expect_type(result$manifest, "list")

  # Cleanup
  unlink(test_dir, recursive = TRUE)
})

test_that("read_simulation returns correct data structure", {
  test_dir <- tempfile()
  df <- create_valid_df(n_households = 5, n_waves = 3)
  manifest <- list(
    model_version = "test",
    country = "test_country"
  )
  create_test_output_dir(test_dir, df, manifest)

  result <- read_simulation(test_dir)

  # Check outcomes
  expect_equal(nrow(result$outcomes), 15)  # 5 households * 3 waves
  expect_equal(length(unique(result$outcomes$household_id)), 5)
  expect_equal(length(unique(result$outcomes$wave)), 3)

  # Check manifest
  expect_equal(result$manifest$model_version, "test")
  expect_equal(result$manifest$country, "test_country")

  # Cleanup
  unlink(test_dir, recursive = TRUE)
})

# ============================================================
# Tests for validation helpers
# ============================================================

test_that("calculate_summaries returns expected structure", {
  df <- create_valid_df()
  summaries <- calculate_summaries(df)

  expect_type(summaries, "list")
  expect_true("by_wave" %in% names(summaries))
  expect_true("by_asset_quintile" %in% names(summaries))
  expect_true("overall" %in% names(summaries))
})

test_that("calculate_summaries handles numeric enterprise_status", {
  df <- create_valid_df() |>
    dplyr::mutate(enterprise_status = as.integer(enterprise_status))

  summaries <- calculate_summaries(df)

  expect_true(all(summaries$by_wave$enterprise_rate >= 0))
  expect_true(all(summaries$by_wave$enterprise_rate <= 1))
})

test_that("run_fe_regression returns fixest object", {
  skip_if_not_installed("fixest")

  df <- create_valid_df(n_households = 100, n_waves = 4)
  model <- run_fe_regression(df)

  expect_s3_class(model, "fixest")
  expect_true("price_exposure" %in% names(coef(model)))
})

test_that("classify_stayers_copers adds classification column", {
  df <- create_valid_df()
  classified <- classify_stayers_copers(df)

  expect_true("classification" %in% names(classified))
  expect_true(all(classified$classification %in% c("stayer", "coper", "none")))
})

test_that("compare_distributions works in toy mode (no observed data)", {
  df <- create_valid_df()
  result <- compare_distributions(df, obs_df = NULL)

  expect_equal(result$mode, "toy")
  expect_null(result$obs_stats)
  expect_null(result$comparison)
  expect_type(result$sim_stats, "list")
})

# ============================================================
# Integration test
# ============================================================

test_that("full pipeline works end-to-end", {
  skip_if_not_installed("fixest")

  # Create test output directory
  test_dir <- tempfile()
  df <- create_valid_df(n_households = 50, n_waves = 4)
  create_test_output_dir(test_dir, df)

  # Read simulation
  sim_data <- read_simulation(test_dir)

  # Calculate summaries
  summaries <- calculate_summaries(sim_data$outcomes)
  expect_type(summaries, "list")

  # Run regression
  model <- run_fe_regression(sim_data$outcomes)
  expect_s3_class(model, "fixest")

  # Compare distributions
  dist_comp <- compare_distributions(sim_data$outcomes)
  expect_equal(dist_comp$mode, "toy")

  # Classify households
  classified <- classify_stayers_copers(sim_data$outcomes)
  expect_true("classification" %in% names(classified))

  # Cleanup
  unlink(test_dir, recursive = TRUE)
})

# Print test summary message
cat("\n=== Test file: test_read_simulation.R ===\n")
cat("Run these tests with: Rscript -e \"testthat::test_file('tests/test_read_simulation.R')\"\n")
cat("Or from analysis directory: make test\n\n")
