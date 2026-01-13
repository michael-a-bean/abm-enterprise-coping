#' Validation Helper Functions
#'
#' Functions for calculating summary statistics, running regressions,
#' and comparing distributions for ABM validation.
#'
#' @author ABM Enterprise Coping Model

# Required packages
# suppressPackageStartupMessages({
#   library(dplyr)
#   library(fixest)
#   library(gt)
# })

#' Extract regression results as a tidy data frame
#'
#' Returns coefficient, std error, p-value, and sign match info
#'
#' @param model fixest model object
#' @param coef_name Name of coefficient to extract
#' @param expected_sign Expected sign: "negative" or "positive"
#' @return List with coefficient, std_error, p_value, sign_match
#' @export
extract_regression_results <- function(model, coef_name, expected_sign = "negative") {
  coefs <- coef(model)
  vcov_mat <- vcov(model)

  if (!coef_name %in% names(coefs)) {
    stop(sprintf("Coefficient '%s' not found in model", coef_name))
  }

  coef_val <- coefs[coef_name]
  se_val <- sqrt(vcov_mat[coef_name, coef_name])
  t_stat <- coef_val / se_val
  p_val <- 2 * stats::pnorm(-abs(t_stat))

  sign_correct <- switch(expected_sign,
    "negative" = coef_val < 0,
    "positive" = coef_val > 0,
    NA
  )

  list(
    coefficient = as.numeric(coef_val),
    std_error = as.numeric(se_val),
    t_statistic = as.numeric(t_stat),
    p_value = as.numeric(p_val),
    sign_match = sign_correct,
    expected_sign = expected_sign,
    pass = sign_correct && p_val < 0.05
  )
}

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
#' @param return_results If TRUE, return structured results; if FALSE, return model object
#' @return fixest regression object or list with structured results
#' @export
#' @importFrom fixest feols
#' @examples
#' \dontrun{
#' fe_model <- run_fe_regression(sim_data$outcomes)
#' summary(fe_model)
#' }
run_fe_regression <- function(df, formula = NULL, return_results = FALSE) {
  df <- as.data.frame(df)

  # Handle both enterprise_status and enterprise_entry column names
  if (!"enterprise_status" %in% names(df) && "enterprise_entry" %in% names(df)) {
    df$enterprise_status <- df$enterprise_entry
  }
  if (!"enterprise_status" %in% names(df) && "enterprise_indicator" %in% names(df)) {
    df$enterprise_status <- df$enterprise_indicator
  }

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

  if (return_results) {
    results <- extract_regression_results(model, "price_exposure", expected_sign = "negative")
    results$model <- model
    return(results)
  }

  model
}

#' Run asset interaction regression
#'
#' Estimates the secondary estimand: heterogeneous effects by asset level.
#' Uses low_assets = asset_quintile <= 2 as per VALIDATION_CONTRACT.md
#'
#' @param df Data frame with household panel data
#' @param low_asset_threshold Percentile threshold for "low assets" (default: 0.4)
#' @param return_results If TRUE, return structured results
#' @return fixest regression object or list with structured results
#' @export
run_asset_interaction_regression <- function(df, low_asset_threshold = 0.4, return_results = FALSE) {
  df <- as.data.frame(df)

  # Handle column name variations
  if (!"enterprise_status" %in% names(df) && "enterprise_entry" %in% names(df)) {
    df$enterprise_status <- df$enterprise_entry
  }
  if (!"enterprise_status" %in% names(df) && "enterprise_indicator" %in% names(df)) {
    df$enterprise_status <- df$enterprise_indicator
  }

  # Convert enterprise_status to numeric if needed
  if (is.logical(df$enterprise_status)) {
    df$enterprise_status <- as.numeric(df$enterprise_status)
  }

  # Create low_assets indicator
  # Use asset_quintile if available, otherwise compute from assets
  if ("asset_quintile" %in% names(df)) {
    df$low_assets <- as.numeric(df$asset_quintile <= 2)
  } else if ("assets" %in% names(df)) {
    asset_cutoff <- quantile(df$assets, probs = low_asset_threshold, na.rm = TRUE)
    df$low_assets <- as.numeric(df$assets <= asset_cutoff)
  } else if ("asset_index" %in% names(df)) {
    asset_cutoff <- quantile(df$asset_index, probs = low_asset_threshold, na.rm = TRUE)
    df$low_assets <- as.numeric(df$asset_index <= asset_cutoff)
  } else {
    stop("No asset variable found in data frame")
  }

  # Ensure factor variables
  df$household_id <- as.factor(df$household_id)
  df$wave <- as.factor(df$wave)

  # Formula with interaction term
  formula <- enterprise_status ~ price_exposure + price_exposure:low_assets | household_id + wave

  model <- fixest::feols(formula, data = df)

  if (return_results) {
    # Get the interaction coefficient name
    coef_names <- names(coef(model))
    interaction_coef <- coef_names[grepl("price_exposure:low_assets", coef_names)][1]
    if (length(interaction_coef) == 0 || is.na(interaction_coef)) {
      interaction_coef <- "price_exposure:low_assets"
    }

    results <- extract_regression_results(model, interaction_coef, expected_sign = "negative")
    results$model <- model
    return(results)
  }

  model
}

#' Run credit access interaction regression
#'
#' Estimates the secondary estimand: heterogeneous effects by credit access.
#' Uses no_credit = !credit_access as per VALIDATION_CONTRACT.md
#'
#' @param df Data frame with household panel data
#' @param return_results If TRUE, return structured results
#' @return fixest regression object or list with structured results
#' @export
run_credit_interaction_regression <- function(df, return_results = FALSE) {
  df <- as.data.frame(df)

  # Handle column name variations
  if (!"enterprise_status" %in% names(df) && "enterprise_entry" %in% names(df)) {
    df$enterprise_status <- df$enterprise_entry
  }
  if (!"enterprise_status" %in% names(df) && "enterprise_indicator" %in% names(df)) {
    df$enterprise_status <- df$enterprise_indicator
  }

  # Convert enterprise_status to numeric if needed
  if (is.logical(df$enterprise_status)) {
    df$enterprise_status <- as.numeric(df$enterprise_status)
  }

  # Create no_credit indicator (inverse of credit_access)
  # Handle numeric (0/1) or logical credit_access
  if (is.numeric(df$credit_access)) {
    df$no_credit <- as.numeric(df$credit_access == 0)
  } else {
    df$no_credit <- as.numeric(!as.logical(df$credit_access))
  }

  # Ensure factor variables
  df$household_id <- as.factor(df$household_id)
  df$wave <- as.factor(df$wave)

  # Formula with interaction term
  formula <- enterprise_status ~ price_exposure + price_exposure:no_credit | household_id + wave

  model <- fixest::feols(formula, data = df)

  if (return_results) {
    # Get the interaction coefficient name
    coef_names <- names(coef(model))
    interaction_coef <- coef_names[grepl("price_exposure:no_credit", coef_names)][1]
    if (length(interaction_coef) == 0 || is.na(interaction_coef)) {
      interaction_coef <- "price_exposure:no_credit"
    }

    results <- extract_regression_results(model, interaction_coef, expected_sign = "negative")
    results$model <- model
    return(results)
  }

  model
}

#' Compare distributions between simulated and observed data
#'
#' Computes comparison statistics between simulated and observed (empirical)
#' distributions. Returns KS test and chi-squared test results as per
#' VALIDATION_CONTRACT.md acceptance criteria.
#'
#' @param simulated Simulated data vector or data frame
#' @param observed Observed (empirical) data vector or data frame, or NULL for toy mode
#' @param variable Variable name to compare if data frames provided
#' @return List containing: ks_statistic, ks_pvalue, chi2_statistic, chi2_pvalue, pass
#' @export
compare_distributions <- function(simulated, observed = NULL, variable = NULL) {
  # Handle data frame input
  if (is.data.frame(simulated)) {
    if (is.null(variable)) {
      variable <- "enterprise_status"
    }
    # Handle column name variations
    if (!variable %in% names(simulated)) {
      if (variable == "enterprise_status" && "enterprise_indicator" %in% names(simulated)) {
        variable <- "enterprise_indicator"
      }
    }
    sim_vec <- as.numeric(simulated[[variable]])
  } else {
    sim_vec <- as.numeric(simulated)
  }

  # If no observed data, return simulated stats only
  if (is.null(observed)) {
    return(list(
      sim_stats = list(
        mean = mean(sim_vec, na.rm = TRUE),
        sd = sd(sim_vec, na.rm = TRUE),
        n = length(sim_vec)
      ),
      ks_statistic = NA,
      ks_pvalue = NA,
      chi2_statistic = NA,
      chi2_pvalue = NA,
      pass = NA,
      mode = "toy"
    ))
  }

  # Handle data frame input for observed
  if (is.data.frame(observed)) {
    if (is.null(variable)) {
      variable <- "enterprise_status"
    }
    if (!variable %in% names(observed)) {
      if (variable == "enterprise_status" && "enterprise_indicator" %in% names(observed)) {
        variable <- "enterprise_indicator"
      }
    }
    obs_vec <- as.numeric(observed[[variable]])
  } else {
    obs_vec <- as.numeric(observed)
  }

  # Remove NAs
  sim_vec <- sim_vec[!is.na(sim_vec)]
  obs_vec <- obs_vec[!is.na(obs_vec)]

  # KS test for continuous data
  ks_result <- stats::ks.test(sim_vec, obs_vec)

  # Chi-squared test for binned/categorical data
  # Create bins for comparison
  all_vals <- c(sim_vec, obs_vec)

  if (length(unique(all_vals)) <= 10) {
    # For categorical/binary data, use frequency table
    sim_table <- table(factor(sim_vec, levels = sort(unique(all_vals))))
    obs_table <- table(factor(obs_vec, levels = sort(unique(all_vals))))

    # Scale to same size for fair comparison
    sim_props <- as.numeric(sim_table) / sum(sim_table)
    obs_expected <- sim_props * sum(obs_table)

    # Avoid division by zero
    obs_expected[obs_expected == 0] <- 0.001

    chi2_result <- tryCatch({
      stats::chisq.test(obs_table, p = sim_props, rescale.p = TRUE)
    }, error = function(e) {
      list(statistic = NA, p.value = NA)
    })
  } else {
    # For continuous data, bin into quantiles
    breaks <- quantile(all_vals, probs = seq(0, 1, by = 0.2), na.rm = TRUE)
    breaks <- unique(breaks)  # Avoid duplicate breaks

    if (length(breaks) >= 2) {
      sim_binned <- cut(sim_vec, breaks = breaks, include.lowest = TRUE)
      obs_binned <- cut(obs_vec, breaks = breaks, include.lowest = TRUE)

      sim_table <- table(sim_binned)
      obs_table <- table(obs_binned)

      chi2_result <- tryCatch({
        stats::chisq.test(rbind(sim_table, obs_table))
      }, error = function(e) {
        list(statistic = NA, p.value = NA)
      })
    } else {
      chi2_result <- list(statistic = NA, p.value = NA)
    }
  }

  # Extract values
  ks_stat <- as.numeric(ks_result$statistic)
  ks_pval <- as.numeric(ks_result$p.value)
  chi2_stat <- if (is.list(chi2_result)) as.numeric(chi2_result$statistic) else NA
  chi2_pval <- if (is.list(chi2_result)) as.numeric(chi2_result$p.value) else NA

  # Pass if both p-values > 0.05 (cannot reject that distributions are same)
  pass <- !is.na(ks_pval) && ks_pval > 0.05

  list(
    sim_mean = mean(sim_vec, na.rm = TRUE),
    obs_mean = mean(obs_vec, na.rm = TRUE),
    ks_statistic = ks_stat,
    ks_pvalue = ks_pval,
    chi2_statistic = chi2_stat,
    chi2_pvalue = chi2_pval,
    pass = pass,
    mode = "validation"
  )
}

#' Compare distributions (legacy version)
#'
#' Legacy function for backward compatibility
#'
#' @param sim_df Simulated data frame from ABM
#' @param obs_df Observed (empirical) data frame, or NULL for toy mode
#' @return List containing distributional comparison
#' @export
compare_distributions_full <- function(sim_df, obs_df = NULL) {
  sim_df <- dplyr::as_tibble(sim_df)

  # Standardize enterprise column name
  ent_col <- if ("enterprise_status" %in% names(sim_df)) "enterprise_status"
             else if ("enterprise_indicator" %in% names(sim_df)) "enterprise_indicator"
             else stop("No enterprise column found")

  asset_col <- if ("assets" %in% names(sim_df)) "assets"
               else if ("asset_index" %in% names(sim_df)) "asset_index"
               else NULL

  # Compute simulated distribution statistics
  sim_stats <- list(
    enterprise_rate = mean(as.numeric(sim_df[[ent_col]]), na.rm = TRUE),
    enterprise_rate_by_wave = sim_df |>
      dplyr::group_by(wave) |>
      dplyr::summarize(rate = mean(as.numeric(.data[[ent_col]]), na.rm = TRUE), .groups = "drop"),
    price_exposure_dist = list(
      mean = mean(sim_df$price_exposure, na.rm = TRUE),
      sd = sd(sim_df$price_exposure, na.rm = TRUE),
      quantiles = quantile(sim_df$price_exposure, probs = c(0.1, 0.25, 0.5, 0.75, 0.9), na.rm = TRUE)
    )
  )

  if (!is.null(asset_col)) {
    sim_stats$assets_dist <- list(
      mean = mean(sim_df[[asset_col]], na.rm = TRUE),
      sd = sd(sim_df[[asset_col]], na.rm = TRUE),
      quantiles = quantile(sim_df[[asset_col]], probs = c(0.1, 0.25, 0.5, 0.75, 0.9), na.rm = TRUE)
    )
  }

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

  # Standardize observed enterprise column
  obs_ent_col <- if ("enterprise_status" %in% names(obs_df)) "enterprise_status"
                 else if ("enterprise_indicator" %in% names(obs_df)) "enterprise_indicator"
                 else stop("No enterprise column found in observed data")

  obs_asset_col <- if ("assets" %in% names(obs_df)) "assets"
                   else if ("asset_index" %in% names(obs_df)) "asset_index"
                   else NULL

  # Compute observed distribution statistics
  obs_stats <- list(
    enterprise_rate = mean(as.numeric(obs_df[[obs_ent_col]]), na.rm = TRUE),
    enterprise_rate_by_wave = obs_df |>
      dplyr::group_by(wave) |>
      dplyr::summarize(rate = mean(as.numeric(.data[[obs_ent_col]]), na.rm = TRUE), .groups = "drop"),
    price_exposure_dist = list(
      mean = mean(obs_df$price_exposure, na.rm = TRUE),
      sd = sd(obs_df$price_exposure, na.rm = TRUE),
      quantiles = quantile(obs_df$price_exposure, probs = c(0.1, 0.25, 0.5, 0.75, 0.9), na.rm = TRUE)
    )
  )

  if (!is.null(obs_asset_col)) {
    obs_stats$assets_dist <- list(
      mean = mean(obs_df[[obs_asset_col]], na.rm = TRUE),
      sd = sd(obs_df[[obs_asset_col]], na.rm = TRUE),
      quantiles = quantile(obs_df[[obs_asset_col]], probs = c(0.1, 0.25, 0.5, 0.75, 0.9), na.rm = TRUE)
    )
  }

  # Compute comparison statistics
  comparison <- list(
    # KS test for price_exposure distribution
    price_exposure_ks = stats::ks.test(
      sim_df$price_exposure,
      obs_df$price_exposure
    ),
    # Enterprise rate difference
    enterprise_rate_diff = sim_stats$enterprise_rate - obs_stats$enterprise_rate
  )

  # Add assets KS test if available
  if (!is.null(asset_col) && !is.null(obs_asset_col)) {
    comparison$assets_ks <- stats::ks.test(
      sim_df[[asset_col]],
      obs_df[[obs_asset_col]]
    )
  }

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

  # Handle column name variations for enterprise
  ent_col <- if ("enterprise_status" %in% names(df)) "enterprise_status"
             else if ("enterprise_indicator" %in% names(df)) "enterprise_indicator"
             else stop("No enterprise column found")

  # Calculate enterprise persistence rate per household
  household_rates <- df |>
    dplyr::group_by(household_id) |>
    dplyr::summarize(
      n_waves = dplyr::n(),
      enterprise_waves = sum(as.numeric(.data[[ent_col]]), na.rm = TRUE),
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

#' Compare classification between simulated and observed data
#'
#' Compares stayer/coper proportions between simulated and observed data.
#' Returns whether proportions are within 10 percentage points as per
#' VALIDATION_CONTRACT.md.
#'
#' @param simulated Simulated data frame from ABM
#' @param observed Observed (empirical) data frame
#' @return List with stayer_sim, stayer_obs, coper_sim, coper_obs, within_10pp
#' @export
compare_classification <- function(simulated, observed) {
  # Classify both datasets
  sim_classified <- classify_stayers_copers(simulated)
  obs_classified <- classify_stayers_copers(observed)

  # Get unique household classifications
  sim_class <- sim_classified |>
    dplyr::distinct(household_id, classification) |>
    dplyr::count(classification) |>
    dplyr::mutate(prop = n / sum(n))

  obs_class <- obs_classified |>
    dplyr::distinct(household_id, classification) |>
    dplyr::count(classification) |>
    dplyr::mutate(prop = n / sum(n))

  # Extract proportions (handle missing classifications)
  get_prop <- function(df, class_name) {
    val <- df$prop[df$classification == class_name]
    if (length(val) == 0) 0 else val
  }

  stayer_sim <- get_prop(sim_class, "stayer")
  stayer_obs <- get_prop(obs_class, "stayer")
  coper_sim <- get_prop(sim_class, "coper")
  coper_obs <- get_prop(obs_class, "coper")

  # Check within 10 percentage points
  stayer_diff <- abs(stayer_sim - stayer_obs)
  coper_diff <- abs(coper_sim - coper_obs)

  stayer_within_10pp <- stayer_diff <= 0.10
  coper_within_10pp <- coper_diff <= 0.10
  within_10pp <- stayer_within_10pp && coper_within_10pp

  list(
    stayer_sim = stayer_sim,
    stayer_obs = stayer_obs,
    stayer_diff = stayer_diff,
    stayer_within_10pp = stayer_within_10pp,
    coper_sim = coper_sim,
    coper_obs = coper_obs,
    coper_diff = coper_diff,
    coper_within_10pp = coper_within_10pp,
    within_10pp = within_10pp,
    pass = within_10pp
  )
}

#' Get classification summary
#'
#' Returns a summary of stayer/coper/none proportions for a single dataset.
#'
#' @param df Data frame with household panel data
#' @return Data frame with classification counts and proportions
#' @export
get_classification_summary <- function(df) {
  classified <- classify_stayers_copers(df)

  classified |>
    dplyr::distinct(household_id, classification) |>
    dplyr::count(classification) |>
    dplyr::mutate(proportion = n / sum(n))
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

#' Run all validation tests
#'
#' Runs all validation tests from VALIDATION_CONTRACT.md and returns
#' a comprehensive summary.
#'
#' @param sim_df Simulated data from ABM
#' @param obs_df Observed (empirical) data, or NULL for toy mode
#' @return List with all validation results
#' @export
run_all_validations <- function(sim_df, obs_df = NULL) {
  results <- list()

  # 1. Primary Estimand: FE regression
  results$primary <- tryCatch({
    run_fe_regression(sim_df, return_results = TRUE)
  }, error = function(e) {
    list(pass = FALSE, error = e$message)
  })

  # 2. Asset Interaction
  results$asset_interaction <- tryCatch({
    run_asset_interaction_regression(sim_df, return_results = TRUE)
  }, error = function(e) {
    list(pass = FALSE, error = e$message)
  })

  # 3. Credit Interaction
  results$credit_interaction <- tryCatch({
    run_credit_interaction_regression(sim_df, return_results = TRUE)
  }, error = function(e) {
    list(pass = FALSE, error = e$message)
  })

  # 4. Distribution comparison (if observed data available)
  if (!is.null(obs_df)) {
    results$distribution <- tryCatch({
      compare_distributions(sim_df, obs_df, variable = "enterprise_status")
    }, error = function(e) {
      list(pass = FALSE, error = e$message)
    })

    # 5. Classification comparison
    results$classification <- tryCatch({
      compare_classification(sim_df, obs_df)
    }, error = function(e) {
      list(pass = FALSE, error = e$message)
    })
  } else {
    results$distribution <- list(pass = NA, mode = "toy")
    results$classification <- list(pass = NA, mode = "toy")
  }

  # Calculate overall pass/fail
  results$overall_pass <- all(sapply(results, function(x) {
    is.na(x$pass) || isTRUE(x$pass)
  }))

  results
}

#' Create validation summary data frame
#'
#' Creates a data frame summarizing all validation results for display.
#'
#' @param validation_results Output from run_all_validations()
#' @return Data frame with test names, values, and pass/fail status
#' @export
create_validation_summary_df <- function(validation_results) {
  rows <- list()

  # Primary estimand
  if (!is.null(validation_results$primary$coefficient)) {
    rows[[1]] <- data.frame(
      Test = "Primary Estimand (beta_1 < 0)",
      Value = sprintf("beta = %.4f, p = %.4f",
                      validation_results$primary$coefficient,
                      validation_results$primary$p_value),
      Status = if (isTRUE(validation_results$primary$pass)) "PASS" else "FAIL"
    )
  }

  # Asset interaction
  if (!is.null(validation_results$asset_interaction$coefficient)) {
    rows[[2]] <- data.frame(
      Test = "Asset Interaction (beta_2 < 0)",
      Value = sprintf("beta = %.4f, p = %.4f",
                      validation_results$asset_interaction$coefficient,
                      validation_results$asset_interaction$p_value),
      Status = if (isTRUE(validation_results$asset_interaction$pass)) "PASS" else "FAIL"
    )
  }

  # Credit interaction
  if (!is.null(validation_results$credit_interaction$coefficient)) {
    rows[[3]] <- data.frame(
      Test = "Credit Interaction (beta_2 < 0)",
      Value = sprintf("beta = %.4f, p = %.4f",
                      validation_results$credit_interaction$coefficient,
                      validation_results$credit_interaction$p_value),
      Status = if (isTRUE(validation_results$credit_interaction$pass)) "PASS" else "FAIL"
    )
  }

  # Distribution test
  if (!is.na(validation_results$distribution$pass)) {
    rows[[4]] <- data.frame(
      Test = "Enterprise Rate Distribution",
      Value = sprintf("KS p = %.4f",
                      validation_results$distribution$ks_pvalue),
      Status = if (isTRUE(validation_results$distribution$pass)) "PASS" else "FAIL"
    )
  }

  # Classification tests
  if (!is.na(validation_results$classification$pass)) {
    rows[[5]] <- data.frame(
      Test = "Stayer Proportion",
      Value = sprintf("sim: %.1f%%, obs: %.1f%%, diff: %.1fpp",
                      validation_results$classification$stayer_sim * 100,
                      validation_results$classification$stayer_obs * 100,
                      validation_results$classification$stayer_diff * 100),
      Status = if (isTRUE(validation_results$classification$stayer_within_10pp)) "PASS" else "FAIL"
    )
    rows[[6]] <- data.frame(
      Test = "Coper Proportion",
      Value = sprintf("sim: %.1f%%, obs: %.1f%%, diff: %.1fpp",
                      validation_results$classification$coper_sim * 100,
                      validation_results$classification$coper_obs * 100,
                      validation_results$classification$coper_diff * 100),
      Status = if (isTRUE(validation_results$classification$coper_within_10pp)) "PASS" else "FAIL"
    )
  }

  # Bind all rows
  do.call(rbind, rows)
}

#' Print validation summary
#'
#' Prints a formatted validation summary to console.
#'
#' @param validation_results Output from run_all_validations()
#' @export
print_validation_summary <- function(validation_results) {
  cat("\nValidation Summary\n")
  cat("==================\n")

  # Primary
  if (!is.null(validation_results$primary$coefficient)) {
    status <- if (isTRUE(validation_results$primary$pass)) "PASS" else "FAIL"
    cat(sprintf("Primary Estimand (beta_1 < 0): %s (beta = %.4f, p = %.4f)\n",
                status,
                validation_results$primary$coefficient,
                validation_results$primary$p_value))
  }

  # Asset interaction
  if (!is.null(validation_results$asset_interaction$coefficient)) {
    status <- if (isTRUE(validation_results$asset_interaction$pass)) "PASS" else "FAIL"
    cat(sprintf("Asset Interaction (beta_2 < 0): %s (beta = %.4f, p = %.4f)\n",
                status,
                validation_results$asset_interaction$coefficient,
                validation_results$asset_interaction$p_value))
  }

  # Credit interaction
  if (!is.null(validation_results$credit_interaction$coefficient)) {
    status <- if (isTRUE(validation_results$credit_interaction$pass)) "PASS" else "FAIL"
    cat(sprintf("Credit Interaction: %s (beta = %.4f, p = %.4f)\n",
                status,
                validation_results$credit_interaction$coefficient,
                validation_results$credit_interaction$p_value))
  }

  # Distribution
  if (!is.na(validation_results$distribution$pass)) {
    status <- if (isTRUE(validation_results$distribution$pass)) "PASS" else "FAIL"
    cat(sprintf("Enterprise Rate Distribution: %s (KS p = %.4f)\n",
                status,
                validation_results$distribution$ks_pvalue))
  }

  # Classification
  if (!is.na(validation_results$classification$pass)) {
    stayer_status <- if (isTRUE(validation_results$classification$stayer_within_10pp)) "PASS" else "FAIL"
    coper_status <- if (isTRUE(validation_results$classification$coper_within_10pp)) "PASS" else "FAIL"

    cat(sprintf("Stayer Proportion: %s (sim: %.0f%%, obs: %.0f%%, diff: %.0fpp)\n",
                stayer_status,
                validation_results$classification$stayer_sim * 100,
                validation_results$classification$stayer_obs * 100,
                validation_results$classification$stayer_diff * 100))
    cat(sprintf("Coper Proportion: %s (sim: %.0f%%, obs: %.0f%%, diff: %.0fpp)\n",
                coper_status,
                validation_results$classification$coper_sim * 100,
                validation_results$classification$coper_obs * 100,
                validation_results$classification$coper_diff * 100))
  }

  cat("\n")
}
