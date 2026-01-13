#' Validation Helper Functions
#'
#' Functions for calculating summary statistics, running regressions,
#' and comparing distributions for ABM validation.
#'
#' @author ABM Enterprise Coping Model

#' Calculate summary statistics for validation
#'
#' Computes key summary statistics from simulation output for validation
#' against empirical patterns.
#'
#' @param df Simulation output data frame with household panel data
#' @return A tibble containing summary statistics:
#'   \itemize{
#'     \item Enterprise rate by wave
#'     \item Enterprise rate by asset quintile
#'     \item Mean price exposure
#'     \item Stayer/coper classification proportions
#'   }
#' @export
#' @importFrom dplyr group_by summarize mutate n ntile
#' @examples
#' \dontrun{
#' summaries <- calculate_summaries(sim_data$outcomes)
#' }
calculate_summaries <- function(df) {
  df <- dplyr::as_tibble(df)

  # Convert enterprise_status to numeric if logical
  if (is.logical(df$enterprise_status)) {
    df$enterprise_status <- as.numeric(df$enterprise_status)
  }

  # Summary 1: Enterprise rate by wave
  by_wave <- df |>
    dplyr::group_by(wave) |>
    dplyr::summarize(
      n_households = dplyr::n(),
      enterprise_rate = mean(enterprise_status, na.rm = TRUE),
      mean_price_exposure = mean(price_exposure, na.rm = TRUE),
      mean_assets = mean(assets, na.rm = TRUE),
      credit_access_rate = mean(as.numeric(credit_access), na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::mutate(summary_type = "by_wave")

  # Summary 2: Enterprise rate by asset quintile
  by_asset_quintile <- df |>
    dplyr::mutate(asset_quintile = dplyr::ntile(assets, 5)) |>
    dplyr::group_by(asset_quintile) |>
    dplyr::summarize(
      n_observations = dplyr::n(),
      enterprise_rate = mean(enterprise_status, na.rm = TRUE),
      mean_price_exposure = mean(price_exposure, na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::mutate(summary_type = "by_asset_quintile")

  # Summary 3: Overall statistics
  overall <- dplyr::tibble(
    statistic = c(
      "total_observations",
      "unique_households",
      "n_waves",
      "overall_enterprise_rate",
      "mean_price_exposure",
      "sd_price_exposure",
      "mean_assets",
      "sd_assets"
    ),
    value = c(
      nrow(df),
      length(unique(df$household_id)),
      length(unique(df$wave)),
      mean(df$enterprise_status, na.rm = TRUE),
      mean(df$price_exposure, na.rm = TRUE),
      sd(df$price_exposure, na.rm = TRUE),
      mean(df$assets, na.rm = TRUE),
      sd(df$assets, na.rm = TRUE)
    )
  )

  # Return as a list for flexibility
  list(
    by_wave = by_wave,
    by_asset_quintile = by_asset_quintile,
    overall = overall
  )
}

#' Run fixed effects regression
#'
#' Estimates the primary estimand: enterprise response to price exposure
#' with household and wave fixed effects.
#'
#' @param df Data frame with household panel data
#' @param formula Optional custom formula (default uses primary estimand)
#' @return fixest regression object
#' @export
#' @importFrom fixest feols
#' @examples
#' \dontrun{
#' fe_model <- run_fe_regression(sim_data$outcomes)
#' summary(fe_model)
#' }
run_fe_regression <- function(df, formula = NULL) {
  df <- as.data.frame(df)

  # Convert enterprise_status to numeric if needed
  if (is.logical(df$enterprise_status)) {
    df$enterprise_status <- as.numeric(df$enterprise_status)
  }

  # Ensure factor variables are properly formatted
  df$household_id <- as.factor(df$household_id)
  df$wave <- as.factor(df$wave)

  # Default formula: primary estimand from VALIDATION_CONTRACT.md
  # enterprise_entry_{it} = beta_1 * price_exposure_{it} + alpha_i + gamma_t + epsilon_{it}
  if (is.null(formula)) {
    formula <- enterprise_status ~ price_exposure | household_id + wave
  }

  # Run fixed effects regression using fixest
  model <- fixest::feols(formula, data = df)

  model
}

#' Run asset interaction regression
#'
#' Estimates the secondary estimand: heterogeneous effects by asset level.
#'
#' @param df Data frame with household panel data
#' @param low_asset_threshold Percentile threshold for "low assets" (default: 0.4)
#' @return fixest regression object
#' @export
run_asset_interaction_regression <- function(df, low_asset_threshold = 0.4) {
  df <- as.data.frame(df)

  # Convert enterprise_status to numeric if needed
  if (is.logical(df$enterprise_status)) {
    df$enterprise_status <- as.numeric(df$enterprise_status)
  }

  # Create low_assets indicator (bottom 40% by default)
  asset_cutoff <- quantile(df$assets, probs = low_asset_threshold, na.rm = TRUE)
  df$low_assets <- as.numeric(df$assets <= asset_cutoff)

  # Ensure factor variables
  df$household_id <- as.factor(df$household_id)
  df$wave <- as.factor(df$wave)

  # Formula with interaction term
  formula <- enterprise_status ~ price_exposure + price_exposure:low_assets | household_id + wave

  fixest::feols(formula, data = df)
}

#' Run credit access interaction regression
#'
#' Estimates the secondary estimand: heterogeneous effects by credit access.
#'
#' @param df Data frame with household panel data
#' @return fixest regression object
#' @export
run_credit_interaction_regression <- function(df) {
  df <- as.data.frame(df)

  # Convert enterprise_status to numeric if needed
  if (is.logical(df$enterprise_status)) {
    df$enterprise_status <- as.numeric(df$enterprise_status)
  }

  # Create no_credit indicator (inverse of credit_access)
  df$no_credit <- as.numeric(!as.logical(df$credit_access))

  # Ensure factor variables
  df$household_id <- as.factor(df$household_id)
  df$wave <- as.factor(df$wave)

  # Formula with interaction term
  formula <- enterprise_status ~ price_exposure + price_exposure:no_credit | household_id + wave

  fixest::feols(formula, data = df)
}

#' Compare distributions between simulated and observed data
#'
#' Computes comparison statistics between simulated and observed (empirical)
#' distributions. If observed data is not provided, returns only simulated
#' distribution statistics.
#'
#' @param sim_df Simulated data frame from ABM
#' @param obs_df Observed (empirical) data frame, or NULL for toy mode
#' @return List containing:
#'   \itemize{
#'     \item sim_stats: Distributional statistics for simulated data
#'     \item obs_stats: Distributional statistics for observed data (if provided)
#'     \item comparison: Comparison tests (KS, chi-squared) if both provided
#'   }
#' @export
compare_distributions <- function(sim_df, obs_df = NULL) {
  sim_df <- dplyr::as_tibble(sim_df)

  # Compute simulated distribution statistics
  sim_stats <- list(
    enterprise_rate = mean(as.numeric(sim_df$enterprise_status), na.rm = TRUE),
    enterprise_rate_by_wave = sim_df |>
      dplyr::group_by(wave) |>
      dplyr::summarize(rate = mean(as.numeric(enterprise_status), na.rm = TRUE), .groups = "drop"),
    price_exposure_dist = list(
      mean = mean(sim_df$price_exposure, na.rm = TRUE),
      sd = sd(sim_df$price_exposure, na.rm = TRUE),
      quantiles = quantile(sim_df$price_exposure, probs = c(0.1, 0.25, 0.5, 0.75, 0.9), na.rm = TRUE)
    ),
    assets_dist = list(
      mean = mean(sim_df$assets, na.rm = TRUE),
      sd = sd(sim_df$assets, na.rm = TRUE),
      quantiles = quantile(sim_df$assets, probs = c(0.1, 0.25, 0.5, 0.75, 0.9), na.rm = TRUE)
    )
  )

  # If no observed data, return simulated stats only
  if (is.null(obs_df)) {
    return(list(
      sim_stats = sim_stats,
      obs_stats = NULL,
      comparison = NULL,
      mode = "toy"
    ))
  }

  obs_df <- dplyr::as_tibble(obs_df)

  # Compute observed distribution statistics
  obs_stats <- list(
    enterprise_rate = mean(as.numeric(obs_df$enterprise_status), na.rm = TRUE),
    enterprise_rate_by_wave = obs_df |>
      dplyr::group_by(wave) |>
      dplyr::summarize(rate = mean(as.numeric(enterprise_status), na.rm = TRUE), .groups = "drop"),
    price_exposure_dist = list(
      mean = mean(obs_df$price_exposure, na.rm = TRUE),
      sd = sd(obs_df$price_exposure, na.rm = TRUE),
      quantiles = quantile(obs_df$price_exposure, probs = c(0.1, 0.25, 0.5, 0.75, 0.9), na.rm = TRUE)
    ),
    assets_dist = list(
      mean = mean(obs_df$assets, na.rm = TRUE),
      sd = sd(obs_df$assets, na.rm = TRUE),
      quantiles = quantile(obs_df$assets, probs = c(0.1, 0.25, 0.5, 0.75, 0.9), na.rm = TRUE)
    )
  )

  # Compute comparison statistics
  comparison <- list(
    # KS test for price_exposure distribution
    price_exposure_ks = stats::ks.test(
      sim_df$price_exposure,
      obs_df$price_exposure
    ),
    # KS test for assets distribution
    assets_ks = stats::ks.test(
      sim_df$assets,
      obs_df$assets
    ),
    # Enterprise rate difference
    enterprise_rate_diff = sim_stats$enterprise_rate - obs_stats$enterprise_rate
  )

  list(
    sim_stats = sim_stats,
    obs_stats = obs_stats,
    comparison = comparison,
    mode = "validation"
  )
}

#' Classify households as stayers or copers
#'
#' Based on the validation contract definition:
#' - Stayer: Operates enterprise in >50% of observed waves
#' - Coper: Operates enterprise in <=50% of observed waves (intermittent)
#'
#' @param df Data frame with household panel data
#' @return Data frame with classification column added
#' @export
classify_stayers_copers <- function(df) {
  df <- dplyr::as_tibble(df)

  # Calculate enterprise persistence rate per household
  household_rates <- df |>
    dplyr::group_by(household_id) |>
    dplyr::summarize(
      n_waves = dplyr::n(),
      enterprise_waves = sum(as.numeric(enterprise_status), na.rm = TRUE),
      enterprise_rate = enterprise_waves / n_waves,
      .groups = "drop"
    ) |>
    dplyr::mutate(
      classification = dplyr::case_when(
        enterprise_rate == 0 ~ "none",
        enterprise_rate > 0.5 ~ "stayer",
        TRUE ~ "coper"
      )
    )

  # Join back to original data
  df |>
    dplyr::left_join(
      household_rates |> dplyr::select(household_id, classification, enterprise_rate),
      by = "household_id",
      suffix = c("", "_calculated")
    )
}

#' Create validation summary table
#'
#' Creates a formatted summary table suitable for reports.
#'
#' @param summaries Output from calculate_summaries()
#' @return gt table object
#' @export
create_summary_table <- function(summaries) {
  summaries$overall |>
    gt::gt() |>
    gt::tab_header(
      title = "Simulation Summary Statistics"
    ) |>
    gt::fmt_number(
      columns = value,
      decimals = 3
    )
}
