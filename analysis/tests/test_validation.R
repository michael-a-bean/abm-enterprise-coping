# Test file for validation_helpers.R
# ABM Enterprise Coping Model
#
# Run with: Rscript -e "testthat::test_file('tests/test_validation.R')"
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

#' Create a minimal valid test data frame with predictable patterns
create_validation_df <- function(n_households = 100, n_waves = 4, seed = 42) {
  set.seed(seed)

  # Create panel structure
  df <- expand.grid(
    household_id = sprintf("HH%03d", 1:n_households),
    wave = 1:n_waves
  ) |>
    dplyr::as_tibble()

  # Generate correlated data where negative price_exposure leads to higher enterprise
  df <- df |>
    dplyr::mutate(
      # Random household-level characteristics
      hh_idx = as.numeric(factor(household_id)),
      base_assets = abs(rnorm(n(), mean = 1000, sd = 500)),
      has_credit = sample(c(TRUE, FALSE), n(), replace = TRUE, prob = c(0.3, 0.7))
    ) |>
    dplyr::group_by(household_id) |>
    dplyr::mutate(
      # Fixed household assets (time-invariant)
      assets = first(base_assets),
      credit_access = first(has_credit),
      # Price exposure varies by wave
      price_exposure = rnorm(n(), mean = -0.05, sd = 0.15)
    ) |>
    dplyr::ungroup() |>
    dplyr::mutate(
      # Enterprise status: more likely when price_exposure is negative
      # This creates the expected negative coefficient
      enterprise_prob = plogis(-0.5 - 1.0 * price_exposure),
      enterprise_status = rbinom(n(), 1, enterprise_prob)
    ) |>
    dplyr::select(
      household_id, wave, enterprise_status, price_exposure,
      assets, credit_access
    )

  df
}

#' Create data with known coefficient signs for testing
create_controlled_df <- function() {
  # Small dataset with predictable pattern:
  # Negative price exposure -> higher enterprise
  tibble::tibble(
    household_id = rep(c("A", "B", "C", "D"), each = 4),
    wave = rep(1:4, 4),
    price_exposure = c(
      -0.2, -0.1, 0.1, 0.2,  # HH A: negative -> enterprise, positive -> no
      -0.3, -0.2, 0.0, 0.1,  # HH B
      0.1, 0.2, -0.1, -0.2,  # HH C: opposite pattern
      0.0, 0.1, -0.1, -0.2   # HH D
    ),
    enterprise_status = c(
      1, 1, 0, 0,  # HH A
      1, 1, 0, 0,  # HH B
      0, 0, 1, 1,  # HH C
      0, 0, 1, 1   # HH D
    ),
    assets = rep(c(500, 800, 1200, 1500), each = 4),
    credit_access = rep(c(FALSE, FALSE, TRUE, TRUE), each = 4)
  )
}

# ============================================================
# Tests for run_fe_regression
# ============================================================

test_that("run_fe_regression returns fixest object", {
  skip_if_not_installed("fixest")

  df <- create_validation_df(n_households = 50, n_waves = 4)
  model <- run_fe_regression(df)

  expect_s3_class(model, "fixest")
  expect_true("price_exposure" %in% names(coef(model)))
})

test_that("run_fe_regression returns correct structure with return_results=TRUE", {
  skip_if_not_installed("fixest")

  df <- create_validation_df(n_households = 50, n_waves = 4)
  results <- run_fe_regression(df, return_results = TRUE)

  expect_type(results, "list")
  expect_true("coefficient" %in% names(results))
  expect_true("std_error" %in% names(results))
  expect_true("p_value" %in% names(results))
  expect_true("sign_match" %in% names(results))
  expect_true("pass" %in% names(results))
  expect_true("model" %in% names(results))

  expect_type(results$coefficient, "double")
  expect_type(results$std_error, "double")
  expect_type(results$p_value, "double")
  expect_type(results$sign_match, "logical")
  expect_type(results$pass, "logical")
})

test_that("run_fe_regression handles enterprise_indicator column name", {
  skip_if_not_installed("fixest")

  df <- create_validation_df(n_households = 50, n_waves = 4) |>
    dplyr::rename(enterprise_indicator = enterprise_status)

  model <- run_fe_regression(df)
  expect_s3_class(model, "fixest")
})

# ============================================================
# Tests for run_asset_interaction_regression
# ============================================================

test_that("run_asset_interaction_regression returns fixest object", {
  skip_if_not_installed("fixest")

  df <- create_validation_df(n_households = 100, n_waves = 4)
  model <- run_asset_interaction_regression(df)

  expect_s3_class(model, "fixest")
  # Should have price_exposure and interaction term
  expect_true(any(grepl("price_exposure", names(coef(model)))))
})

test_that("run_asset_interaction_regression handles asset_quintile column", {
  skip_if_not_installed("fixest")

  df <- create_validation_df(n_households = 100, n_waves = 4) |>
    dplyr::mutate(asset_quintile = ntile(assets, 5))

  results <- run_asset_interaction_regression(df, return_results = TRUE)

  expect_type(results, "list")
  expect_true("coefficient" %in% names(results))
})

# ============================================================
# Tests for run_credit_interaction_regression
# ============================================================

test_that("run_credit_interaction_regression returns fixest object", {
  skip_if_not_installed("fixest")

  df <- create_validation_df(n_households = 100, n_waves = 4)
  model <- run_credit_interaction_regression(df)

  expect_s3_class(model, "fixest")
})

test_that("run_credit_interaction_regression returns results structure", {
  skip_if_not_installed("fixest")

  df <- create_validation_df(n_households = 100, n_waves = 4)
  results <- run_credit_interaction_regression(df, return_results = TRUE)

  expect_type(results, "list")
  expect_true("coefficient" %in% names(results))
  expect_true("pass" %in% names(results))
})

# ============================================================
# Tests for compare_distributions
# ============================================================

test_that("compare_distributions works with data frames", {
  df1 <- create_validation_df(n_households = 50, n_waves = 4, seed = 1)
  df2 <- create_validation_df(n_households = 50, n_waves = 4, seed = 2)

  result <- compare_distributions(df1, df2, variable = "enterprise_status")

  expect_type(result, "list")
  expect_true("ks_statistic" %in% names(result))
  expect_true("ks_pvalue" %in% names(result))
  expect_true("chi2_statistic" %in% names(result))
  expect_true("chi2_pvalue" %in% names(result))
  expect_true("pass" %in% names(result))
  expect_equal(result$mode, "validation")
})

test_that("compare_distributions returns toy mode when no observed data", {
  df <- create_validation_df(n_households = 50, n_waves = 4)

  result <- compare_distributions(df, NULL, variable = "enterprise_status")

  expect_equal(result$mode, "toy")
  expect_true(is.na(result$ks_statistic))
  expect_true(is.na(result$pass))
})

test_that("compare_distributions works with vectors", {
  vec1 <- rnorm(100)
  vec2 <- rnorm(100)

  result <- compare_distributions(vec1, vec2)

  expect_type(result, "list")
  expect_true(!is.na(result$ks_statistic))
  expect_true(!is.na(result$ks_pvalue))
})

test_that("compare_distributions passes when distributions are similar", {
  # Same distribution should pass
  set.seed(123)
  vec1 <- rbinom(1000, 1, 0.3)
  vec2 <- rbinom(1000, 1, 0.3)

  result <- compare_distributions(vec1, vec2)

  # Should likely pass (high p-value) when drawn from same distribution
  expect_true(result$ks_pvalue > 0.01)  # Relaxed threshold for stochastic test
})

# ============================================================
# Tests for compare_classification
# ============================================================

test_that("compare_classification returns correct structure", {
  df1 <- create_validation_df(n_households = 50, n_waves = 4, seed = 1)
  df2 <- create_validation_df(n_households = 50, n_waves = 4, seed = 2)

  result <- compare_classification(df1, df2)

  expect_type(result, "list")
  expect_true("stayer_sim" %in% names(result))
  expect_true("stayer_obs" %in% names(result))
  expect_true("coper_sim" %in% names(result))
  expect_true("coper_obs" %in% names(result))
  expect_true("within_10pp" %in% names(result))
  expect_true("pass" %in% names(result))
})

test_that("compare_classification computes proportions correctly", {
  # Create data where we know the classification
  # Stayer: >50% enterprise waves
  # Coper: 1-50% enterprise waves
  # None: 0% enterprise waves

  df <- tibble::tibble(
    household_id = rep(c("stayer1", "stayer2", "coper1", "none1"), each = 4),
    wave = rep(1:4, 4),
    enterprise_status = c(
      1, 1, 1, 1,  # stayer1: 100%
      1, 1, 1, 0,  # stayer2: 75%
      1, 1, 0, 0,  # coper1: 50% (coper)
      0, 0, 0, 0   # none1: 0%
    ),
    price_exposure = 0,
    assets = 1000,
    credit_access = TRUE
  )

  result <- compare_classification(df, df)

  # Same data should be within 10pp
  expect_true(result$within_10pp)
  expect_equal(result$stayer_diff, 0)
  expect_equal(result$coper_diff, 0)
})

test_that("compare_classification detects differences > 10pp", {
  # Create two datasets with very different classifications
  df1 <- tibble::tibble(
    household_id = rep(c("A", "B"), each = 4),
    wave = rep(1:4, 2),
    enterprise_status = c(1, 1, 1, 1, 0, 0, 0, 0),  # 50% stayer, 50% none
    price_exposure = 0,
    assets = 1000,
    credit_access = TRUE
  )

  df2 <- tibble::tibble(
    household_id = rep(c("A", "B"), each = 4),
    wave = rep(1:4, 2),
    enterprise_status = c(0, 0, 0, 0, 0, 0, 0, 0),  # 100% none
    price_exposure = 0,
    assets = 1000,
    credit_access = TRUE
  )

  result <- compare_classification(df1, df2)

  # Should detect the 50pp difference
  expect_false(result$within_10pp)
})

# ============================================================
# Tests for classify_stayers_copers
# ============================================================

test_that("classify_stayers_copers adds classification column", {
  df <- create_validation_df(n_households = 50, n_waves = 4)
  result <- classify_stayers_copers(df)

  expect_true("classification" %in% names(result))
  expect_true(all(result$classification %in% c("stayer", "coper", "none")))
})

test_that("classify_stayers_copers correctly classifies households", {
  df <- tibble::tibble(
    household_id = rep(c("stayer", "coper", "none"), each = 4),
    wave = rep(1:4, 3),
    enterprise_status = c(
      1, 1, 1, 0,  # stayer: 75% > 50%
      1, 0, 0, 0,  # coper: 25% <= 50%
      0, 0, 0, 0   # none: 0%
    ),
    price_exposure = 0,
    assets = 1000,
    credit_access = TRUE
  )

  result <- classify_stayers_copers(df)

  stayer_class <- unique(result$classification[result$household_id == "stayer"])
  coper_class <- unique(result$classification[result$household_id == "coper"])
  none_class <- unique(result$classification[result$household_id == "none"])

  expect_equal(stayer_class, "stayer")
  expect_equal(coper_class, "coper")
  expect_equal(none_class, "none")
})

# ============================================================
# Tests for get_classification_summary
# ============================================================

test_that("get_classification_summary returns correct structure", {
  df <- create_validation_df(n_households = 50, n_waves = 4)
  result <- get_classification_summary(df)

  expect_s3_class(result, "data.frame")
  expect_true("classification" %in% names(result))
  expect_true("n" %in% names(result))
  expect_true("proportion" %in% names(result))
})

test_that("get_classification_summary proportions sum to 1", {
  df <- create_validation_df(n_households = 50, n_waves = 4)
  result <- get_classification_summary(df)

  expect_equal(sum(result$proportion), 1, tolerance = 1e-10)
})

# ============================================================
# Tests for extract_regression_results
# ============================================================

test_that("extract_regression_results works with fixest model", {
  skip_if_not_installed("fixest")

  df <- create_validation_df(n_households = 100, n_waves = 4)
  model <- run_fe_regression(df)

  result <- extract_regression_results(model, "price_exposure", expected_sign = "negative")

  expect_type(result, "list")
  expect_true("coefficient" %in% names(result))
  expect_true("std_error" %in% names(result))
  expect_true("p_value" %in% names(result))
  expect_true("sign_match" %in% names(result))
  expect_true("pass" %in% names(result))
})

test_that("extract_regression_results errors for missing coefficient", {
  skip_if_not_installed("fixest")

  df <- create_validation_df(n_households = 100, n_waves = 4)
  model <- run_fe_regression(df)

  expect_error(
    extract_regression_results(model, "nonexistent_coef"),
    "not found in model"
  )
})

# ============================================================
# Tests for run_all_validations
# ============================================================

test_that("run_all_validations runs without error in toy mode", {
  skip_if_not_installed("fixest")

  df <- create_validation_df(n_households = 100, n_waves = 4)
  result <- run_all_validations(df, obs_df = NULL)

  expect_type(result, "list")
  expect_true("primary" %in% names(result))
  expect_true("asset_interaction" %in% names(result))
  expect_true("credit_interaction" %in% names(result))
  expect_true("distribution" %in% names(result))
  expect_true("classification" %in% names(result))
  expect_true("overall_pass" %in% names(result))
})

test_that("run_all_validations runs with observed data", {
  skip_if_not_installed("fixest")

  df1 <- create_validation_df(n_households = 100, n_waves = 4, seed = 1)
  df2 <- create_validation_df(n_households = 100, n_waves = 4, seed = 2)

  result <- run_all_validations(df1, obs_df = df2)

  expect_type(result, "list")
  expect_equal(result$distribution$mode, "validation")
  expect_type(result$classification$pass, "logical")
})

# ============================================================
# Integration test
# ============================================================

test_that("full validation pipeline runs end-to-end", {
  skip_if_not_installed("fixest")

  # Create test data
  df <- create_validation_df(n_households = 100, n_waves = 4)

  # Run FE regression
  fe_results <- run_fe_regression(df, return_results = TRUE)
  expect_type(fe_results$coefficient, "double")

  # Run asset interaction
  asset_results <- run_asset_interaction_regression(df, return_results = TRUE)
  expect_type(asset_results$coefficient, "double")

  # Run credit interaction
  credit_results <- run_credit_interaction_regression(df, return_results = TRUE)
  expect_type(credit_results$coefficient, "double")

  # Run distribution comparison (toy mode)
  dist_results <- compare_distributions(df, NULL)
  expect_equal(dist_results$mode, "toy")

  # Get classification summary
  class_summary <- get_classification_summary(df)
  expect_equal(sum(class_summary$proportion), 1, tolerance = 1e-10)

  # Run all validations
  all_results <- run_all_validations(df)
  expect_type(all_results$overall_pass, "logical")
})

# Print test summary message
cat("\n=== Test file: test_validation.R ===\n")
cat("Run these tests with: Rscript -e \"testthat::test_file('tests/test_validation.R')\"\n")
cat("Or from analysis directory: make test\n\n")
