#' Analysis Helper Functions for ABM Reports
#'
#' Provides additional utilities for batch analysis, sensitivity analysis,
#' phase portraits, and heatmaps.
#'
#' @author ABM Enterprise Coping Model

library(dplyr)
library(tidyr)
library(purrr)
library(arrow)
library(ggplot2)

# Load core theme
source(here::here("R", "plot_theme.R"))

#' Read batch simulation results
#'
#' Loads multiple simulation runs from a batch directory.
#'
#' @param batch_dir Directory containing seed_* subdirectories
#' @return Data frame with all outcomes and seed identifier
#' @export
read_batch_simulations <- function(batch_dir) {
  # Find all seed directories
  seed_dirs <- list.dirs(batch_dir, recursive = FALSE, full.names = TRUE)
  seed_dirs <- seed_dirs[grepl("seed_", basename(seed_dirs))]

  if (length(seed_dirs) == 0) {
    stop("No seed directories found in: ", batch_dir)
  }

  # Read each simulation
  results <- lapply(seed_dirs, function(d) {
    outcomes_path <- file.path(d, "household_outcomes.parquet")
    manifest_path <- file.path(d, "manifest.json")

    if (!file.exists(manifest_path) && !dir.exists(outcomes_path)) {
      return(NULL)
    }

    # Read outcomes (handle partitioned or single file)
    if (dir.exists(outcomes_path)) {
      df <- arrow::open_dataset(outcomes_path) |> collect()
    } else if (file.exists(outcomes_path)) {
      df <- arrow::read_parquet(outcomes_path)
    } else {
      return(NULL)
    }

    # Extract seed from directory name
    seed <- as.integer(sub("seed_", "", basename(d)))
    df$seed <- seed
    df
  })

  # Combine all results
  bind_rows(results[!sapply(results, is.null)])
}

#' Calculate robustness metrics across seeds
#'
#' Computes mean, SD, and CV of key metrics across replicate runs.
#'
#' @param batch_df Data frame from read_batch_simulations()
#' @return Data frame with robustness metrics
#' @export
calculate_robustness_metrics <- function(batch_df) {
  # Enterprise rate per seed
  by_seed <- batch_df |>
    group_by(seed) |>
    summarize(
      enterprise_rate = mean(as.numeric(enterprise_status), na.rm = TRUE),
      mean_assets = mean(assets_index, na.rm = TRUE),
      mean_price_exposure = mean(price_exposure, na.rm = TRUE),
      n_households = n_distinct(household_id),
      n_waves = n_distinct(wave),
      .groups = "drop"
    )

  # Aggregate across seeds
  robustness <- tibble(
    metric = c("enterprise_rate", "mean_assets", "mean_price_exposure"),
    mean = c(
      mean(by_seed$enterprise_rate),
      mean(by_seed$mean_assets),
      mean(by_seed$mean_price_exposure)
    ),
    sd = c(
      sd(by_seed$enterprise_rate),
      sd(by_seed$mean_assets),
      sd(by_seed$mean_price_exposure)
    )
  ) |>
    mutate(
      cv = sd / abs(mean),
      n_seeds = nrow(by_seed)
    )

  list(
    by_seed = by_seed,
    summary = robustness
  )
}

#' Plot enterprise rate distribution across seeds
#'
#' Creates a violin or jitter plot showing variation across replicates.
#'
#' @param batch_df Data frame from read_batch_simulations()
#' @return ggplot object
#' @export
plot_enterprise_rate_robustness <- function(batch_df) {
  by_seed <- batch_df |>
    group_by(seed) |>
    summarize(
      enterprise_rate = mean(as.numeric(enterprise_status), na.rm = TRUE),
      .groups = "drop"
    )

  overall_mean <- mean(by_seed$enterprise_rate)
  overall_sd <- sd(by_seed$enterprise_rate)

  ggplot(by_seed, aes(x = 1, y = enterprise_rate)) +
    geom_jitter(width = 0.1, height = 0, alpha = 0.6, size = 2.5, color = "#0072B2") +
    geom_hline_ref(yintercept = overall_mean) +
    annotate(
      "rect",
      xmin = 0.5, xmax = 1.5,
      ymin = overall_mean - overall_sd,
      ymax = overall_mean + overall_sd,
      alpha = 0.2, fill = "grey60"
    ) +
    scale_y_continuous(labels = scales::percent_format()) +
    labs(
      x = NULL,
      y = "Enterprise Rate"
    ) +
    theme_abm_minimal() +
    theme(
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank()
    )
}

#' Create phase portrait from simulation data
#'
#' Plots trajectories through state space across waves.
#'
#' @param df Simulation data frame
#' @param x_var Variable for x-axis (default: "mean_price_exposure")
#' @param y_var Variable for y-axis (default: "enterprise_rate")
#' @return ggplot object
#' @export
create_phase_portrait <- function(
    df,
    x_var = "mean_price_exposure",
    y_var = "enterprise_rate"
) {
  # Aggregate by wave
  phase_data <- df |>
    group_by(wave) |>
    summarize(
      enterprise_rate = mean(as.numeric(enterprise_status), na.rm = TRUE),
      mean_price_exposure = mean(price_exposure, na.rm = TRUE),
      mean_assets = mean(assets_index, na.rm = TRUE),
      .groups = "drop"
    ) |>
    arrange(wave)

  ggplot(phase_data, aes(x = .data[[x_var]], y = .data[[y_var]])) +
    geom_path(color = "grey50", linewidth = 0.8, arrow = arrow(
      length = unit(0.15, "inches"),
      type = "closed"
    )) +
    geom_point(aes(color = factor(wave)), size = 4) +
    geom_text(
      aes(label = paste0("W", wave)),
      nudge_y = 0.015,
      size = 3,
      fontface = "bold"
    ) +
    scale_color_abm() +
    scale_y_continuous(labels = scales::percent_format()) +
    labs(
      x = "Mean Price Exposure",
      y = "Enterprise Rate",
      color = "Wave"
    ) +
    theme_abm_minimal() +
    theme(legend.position = "none")
}

#' Create multi-seed phase portrait
#'
#' Overlays phase trajectories from multiple seeds.
#'
#' @param batch_df Data frame from read_batch_simulations()
#' @return ggplot object
#' @export
create_multi_seed_phase_portrait <- function(batch_df) {
  # Aggregate by seed and wave
  phase_data <- batch_df |>
    group_by(seed, wave) |>
    summarize(
      enterprise_rate = mean(as.numeric(enterprise_status), na.rm = TRUE),
      mean_price_exposure = mean(price_exposure, na.rm = TRUE),
      .groups = "drop"
    ) |>
    arrange(seed, wave)

  # Mean trajectory
  mean_trajectory <- phase_data |>
    group_by(wave) |>
    summarize(
      enterprise_rate = mean(enterprise_rate),
      mean_price_exposure = mean(mean_price_exposure),
      .groups = "drop"
    )

  ggplot() +
    # Individual seed trajectories (faint)
    geom_path(
      data = phase_data,
      aes(x = mean_price_exposure, y = enterprise_rate, group = seed),
      color = "grey80",
      linewidth = 0.3,
      alpha = 0.5
    ) +
    # Mean trajectory (bold)
    geom_path(
      data = mean_trajectory,
      aes(x = mean_price_exposure, y = enterprise_rate),
      color = "#0072B2",
      linewidth = 1.2,
      arrow = arrow(length = unit(0.15, "inches"), type = "closed")
    ) +
    geom_point(
      data = mean_trajectory,
      aes(x = mean_price_exposure, y = enterprise_rate, color = factor(wave)),
      size = 4
    ) +
    geom_text(
      data = mean_trajectory,
      aes(x = mean_price_exposure, y = enterprise_rate, label = paste0("W", wave)),
      nudge_y = 0.015,
      size = 3,
      fontface = "bold"
    ) +
    scale_color_abm() +
    scale_y_continuous(labels = scales::percent_format()) +
    labs(
      x = "Mean Price Exposure",
      y = "Enterprise Rate",
      color = "Wave"
    ) +
    theme_abm_minimal() +
    theme(legend.position = "none")
}

#' Generate parameter sweep grid
#'
#' Creates a parameter grid for sensitivity analysis.
#'
#' @param param1_name Name of first parameter
#' @param param1_range Range of values for first parameter
#' @param param2_name Name of second parameter
#' @param param2_range Range of values for second parameter
#' @return Data frame with parameter combinations
#' @export
generate_sweep_grid <- function(
    param1_name,
    param1_range,
    param2_name,
    param2_range
) {
  grid <- expand.grid(
    p1 = param1_range,
    p2 = param2_range
  )
  names(grid) <- c(param1_name, param2_name)
  grid
}

#' Create sensitivity heatmap
#'
#' Generates a heatmap from sweep results.
#'
#' @param sweep_df Data frame with sweep results
#' @param x_var Parameter on x-axis
#' @param y_var Parameter on y-axis
#' @param fill_var Outcome variable for fill color
#' @param diverging Use diverging color scale (default: FALSE)
#' @param midpoint Midpoint for diverging scale
#' @return ggplot object
#' @export
create_sensitivity_heatmap <- function(
    sweep_df,
    x_var,
    y_var,
    fill_var,
    diverging = FALSE,
    midpoint = NULL
) {
  p <- ggplot(sweep_df, aes(x = .data[[x_var]], y = .data[[y_var]], fill = .data[[fill_var]])) +
    geom_tile() +
    labs(
      x = x_var,
      y = y_var,
      fill = fill_var
    ) +
    theme_abm_minimal() +
    theme(legend.position = "right")

  if (diverging && !is.null(midpoint)) {
    p <- p + scale_fill_abm_diverging(midpoint = midpoint)
  } else {
    p <- p + scale_fill_abm_seq()
  }

  p
}

#' Compute standardized coefficients from fixest model
#'
#' @param model fixest model object
#' @param df Original data frame
#' @return Data frame with standardized coefficients
#' @export
standardized_coefficients <- function(model, df) {
  coefs <- coef(model)
  vcov_mat <- vcov(model)

  result <- data.frame(
    term = names(coefs),
    estimate = as.numeric(coefs),
    std_error = sqrt(diag(vcov_mat))
  )

  # Get SDs of variables
  for (i in seq_len(nrow(result))) {
    term <- result$term[i]
    if (term %in% names(df)) {
      x_sd <- sd(df[[term]], na.rm = TRUE)
      y_sd <- sd(df$enterprise_status, na.rm = TRUE)
      result$std_estimate[i] <- result$estimate[i] * (x_sd / y_sd)
    } else {
      result$std_estimate[i] <- NA
    }
  }

  result
}

#' Create coefficient plot
#'
#' Visualizes regression coefficients with confidence intervals.
#'
#' @param model fixest model object
#' @param significance_level Significance level for CI (default: 0.05)
#' @return ggplot object
#' @export
plot_coefficients <- function(model, significance_level = 0.05) {
  coefs <- coef(model)
  se <- sqrt(diag(vcov(model)))
  z <- qnorm(1 - significance_level / 2)

  coef_df <- data.frame(
    term = names(coefs),
    estimate = as.numeric(coefs),
    lower = as.numeric(coefs - z * se),
    upper = as.numeric(coefs + z * se),
    significant = abs(coefs / se) > z
  )

  ggplot(coef_df, aes(x = estimate, y = reorder(term, estimate))) +
    geom_vline_ref(xintercept = 0) +
    geom_errorbarh(
      aes(xmin = lower, xmax = upper),
      height = 0.2,
      color = "grey40"
    ) +
    geom_point(
      aes(color = significant),
      size = 3
    ) +
    scale_color_manual(
      values = c("FALSE" = "grey60", "TRUE" = "#0072B2"),
      guide = "none"
    ) +
    labs(
      x = "Coefficient Estimate",
      y = NULL
    ) +
    theme_abm_minimal()
}

#' Calculate transition rates by wave
#'
#' @param df Simulation data frame
#' @return Data frame with transition rates per wave
#' @export
calculate_transition_rates <- function(df) {
  df <- df |>
    arrange(household_id, wave) |>
    group_by(household_id) |>
    mutate(
      prev_status = lag(enterprise_status),
      transition = case_when(
        is.na(prev_status) ~ NA_character_,
        prev_status == 0 & enterprise_status == 1 ~ "ENTER",
        prev_status == 1 & enterprise_status == 0 ~ "EXIT",
        TRUE ~ "STAY"
      )
    ) |>
    ungroup()

  df |>
    filter(!is.na(transition)) |>
    group_by(wave) |>
    summarize(
      n_transitions = n(),
      enter_rate = mean(transition == "ENTER"),
      exit_rate = mean(transition == "EXIT"),
      stay_rate = mean(transition == "STAY"),
      .groups = "drop"
    )
}

#' Plot transition rates over time
#'
#' @param df Simulation data frame
#' @return ggplot object
#' @export
plot_transition_rates <- function(df) {
  rates <- calculate_transition_rates(df)

  rates_long <- rates |>
    select(wave, enter_rate, exit_rate) |>
    pivot_longer(
      cols = c(enter_rate, exit_rate),
      names_to = "transition_type",
      values_to = "rate"
    ) |>
    mutate(
      transition_type = case_when(
        transition_type == "enter_rate" ~ "Entry",
        transition_type == "exit_rate" ~ "Exit"
      )
    )

  ggplot(rates_long, aes(x = factor(wave), y = rate, fill = transition_type)) +
    geom_col(position = "dodge", alpha = 0.8) +
    scale_fill_manual(values = c("Entry" = "#009E73", "Exit" = "#CC79A7")) +
    scale_y_continuous(labels = scales::percent_format()) +
    labs(
      x = "Wave",
      y = "Transition Rate",
      fill = "Transition"
    ) +
    theme_abm_minimal()
}

#' Classification breakdown table
#'
#' Creates a summary table of stayer/coper/none proportions.
#'
#' @param df Simulation data frame with classification column
#' @return Data frame with counts and proportions
#' @export
classification_summary <- function(df) {
  if (!"classification" %in% names(df)) {
    stop("Data frame must have 'classification' column")
  }

  df |>
    distinct(household_id, classification) |>
    count(classification) |>
    mutate(
      proportion = n / sum(n),
      pct = scales::percent(proportion, accuracy = 0.1)
    )
}

#' Compare classification between simulated and observed
#'
#' @param sim_df Simulated data
#' @param obs_df Observed data
#' @return Comparison data frame
#' @export
compare_classification <- function(sim_df, obs_df) {
  sim_class <- classification_summary(sim_df) |>
    select(classification, sim_n = n, sim_pct = pct)

  obs_class <- classification_summary(obs_df) |>
    select(classification, obs_n = n, obs_pct = pct)

  full_join(sim_class, obs_class, by = "classification")
}
