#' Minimalist Publication Theme for ABM Reports
#'
#' Provides a consistent, publication-ready ggplot2 theme for all figures
#' in the ABM Enterprise Coping Model reports.
#'
#' Design principles (per AI-sourced guidance, 2026-01-13):
#' - No plot titles (use figure captions instead)
#' - Minimal gridlines (major only, subtle grey)
#' - Black/grey reference lines (no red)
#' - Clear axis labels with units
#' - Colorblind-safe palettes (Okabe-Ito, viridis)
#' - Consistent typography
#'
#' @author ABM Enterprise Coping Model
#' @references
#'   - Tufte, E. (2001). The Visual Display of Quantitative Information
#'   - Wickham, H. (2016). ggplot2: Elegant graphics for data analysis
#'   - AI-sourced: OpenAI o1, Gemini deep-research (2026-01-13)

library(ggplot2)
library(scales)

#' Theme for ABM publication-quality figures
#'
#' A minimalist theme that removes clutter and emphasizes data.
#' Designed to work well in both HTML and PDF output.
#'
#' @param base_size Base font size (default: 11)
#' @param base_family Base font family (default: "sans")
#' @return A ggplot2 theme object
#' @export
#' @examples
#' ggplot(mtcars, aes(wt, mpg)) +
#'   geom_point() +
#'   theme_abm_minimal()
theme_abm_minimal <- function(base_size = 11, base_family = "sans") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      # Remove plot title (use fig-cap in Quarto instead)
      plot.title = element_blank(),
      plot.subtitle = element_text(
        color = "grey30",
        size = rel(0.9),
        margin = margin(b = 10)
      ),

      # Caption styling for data source notes
      plot.caption = element_text(
        color = "grey50",
        size = rel(0.7),
        hjust = 1,
        margin = margin(t = 10)
      ),

      # Major gridlines only, subtle
      panel.grid.major = element_line(
        color = "grey92",
        linewidth = 0.3
      ),
      panel.grid.minor = element_blank(),

      # Clear axis lines
      axis.line = element_line(
        color = "black",
        linewidth = 0.4
      ),
      axis.ticks = element_line(
        color = "black",
        linewidth = 0.3
      ),

      # Axis text styling
      axis.title = element_text(
        size = rel(0.95),
        color = "grey20"
      ),
      axis.text = element_text(
        size = rel(0.85),
        color = "grey30"
      ),

      # Legend styling
      legend.position = "bottom",
      legend.title = element_text(
        face = "bold",
        size = rel(0.9)
      ),
      legend.text = element_text(
        size = rel(0.85)
      ),
      legend.key.size = unit(0.8, "lines"),
      legend.margin = margin(t = 5),

      # Panel and strip styling
      strip.text = element_text(
        face = "bold",
        size = rel(0.9),
        margin = margin(b = 5)
      ),
      strip.background = element_blank(),

      # Overall plot margins
      plot.margin = margin(10, 10, 10, 10)
    )
}

#' Colorblind-safe discrete color palette
#'
#' Based on Okabe-Ito palette, excluding red for reference lines.
#'
#' @param n Number of colors needed
#' @return Character vector of hex colors
#' @export
palette_abm_discrete <- function(n = 8) {
  # Okabe-Ito palette (colorblind-safe)
  colors <- c(
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#009E73",  # Green
    "#CC79A7",  # Pink
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#999999"
  )
  if (n > length(colors)) {
    warning("Requested more colors than available in palette")
  }
  colors[seq_len(min(n, length(colors)))]
}

#' Discrete color scale for ABM figures
#'
#' @param ... Additional arguments passed to scale_color_manual
#' @return A ggplot2 scale object
#' @export
scale_color_abm <- function(...) {
  scale_color_manual(values = palette_abm_discrete(8), ...)
}

#' Discrete fill scale for ABM figures
#'
#' @param ... Additional arguments passed to scale_fill_manual
#' @return A ggplot2 scale object
#' @export
scale_fill_abm <- function(...) {
  scale_fill_manual(values = palette_abm_discrete(8), ...)
}

#' Sequential color scale (for heatmaps, continuous variables)
#'
#' Uses viridis for perceptual uniformity and colorblind safety.
#'
#' @param option Viridis palette option (default: "D")
#' @param ... Additional arguments passed to scale_*_viridis_c
#' @return A ggplot2 scale object
#' @export
scale_color_abm_seq <- function(option = "D", ...) {
  scale_color_viridis_c(option = option, ...)
}

#' Sequential fill scale (for heatmaps)
#'
#' @param option Viridis palette option (default: "D")
#' @param ... Additional arguments passed to scale_fill_viridis_c
#' @return A ggplot2 scale object
#' @export
scale_fill_abm_seq <- function(option = "D", ...) {
  scale_fill_viridis_c(option = option, ...)
}

#' Diverging color scale (for deviation from baseline)
#'
#' Blue-white-orange diverging scale centered at zero.
#'
#' @param midpoint Midpoint value (default: 0)
#' @param low Low color (default: "#0072B2" blue)
#' @param mid Mid color (default: "white")
#' @param high High color (default: "#E69F00" orange)
#' @param ... Additional arguments
#' @return A ggplot2 scale object
#' @export
scale_fill_abm_diverging <- function(
    midpoint = 0,
    low = "#0072B2",
    mid = "white",
    high = "#E69F00",
    ...
) {
  scale_fill_gradient2(
    low = low,
    mid = mid,
    high = high,
    midpoint = midpoint,
    ...
  )
}

#' Add horizontal reference line (black, dashed)
#'
#' Consistent styling for reference lines across figures.
#'
#' @param yintercept Y-axis intercept value
#' @param linetype Line type (default: "dashed")
#' @param color Line color (default: "grey40")
#' @param linewidth Line width (default: 0.5)
#' @return A ggplot2 geom object
#' @export
geom_hline_ref <- function(
    yintercept,
    linetype = "dashed",
    color = "grey40",
    linewidth = 0.5
) {
  geom_hline(
    yintercept = yintercept,
    linetype = linetype,
    color = color,
    linewidth = linewidth
  )
}

#' Add vertical reference line (black, dashed)
#'
#' @param xintercept X-axis intercept value
#' @param linetype Line type (default: "dashed")
#' @param color Line color (default: "grey40")
#' @param linewidth Line width (default: 0.5)
#' @return A ggplot2 geom object
#' @export
geom_vline_ref <- function(
    xintercept,
    linetype = "dashed",
    color = "grey40",
    linewidth = 0.5
) {
  geom_vline(
    xintercept = xintercept,
    linetype = linetype,
    color = color,
    linewidth = linewidth
  )
}

#' Format numbers for publication (thousands separator, decimal control)
#'
#' @param x Numeric vector
#' @param digits Number of decimal places
#' @return Formatted character vector
#' @export
fmt_num <- function(x, digits = 2) {
  format(round(x, digits), big.mark = ",", nsmall = digits)
}

#' Format p-values for publication
#'
#' @param p P-value
#' @param digits Significant digits for small p-values
#' @return Formatted character string
#' @export
fmt_pvalue <- function(p, digits = 3) {
  if (p < 0.001) {
    return("< 0.001")
  } else if (p < 0.01) {
    return(sprintf("%.3f", p))
  } else {
    return(sprintf("%.2f", p))
  }
}

#' Format coefficient with standard error
#'
#' @param coef Coefficient estimate
#' @param se Standard error
#' @param digits Decimal places
#' @return Formatted string "coef (se)"
#' @export
fmt_coef_se <- function(coef, se, digits = 3) {
  sprintf("%.*f (%.*f)", digits, coef, digits, se)
}

#' Classification colors for stayer/coper/none
#'
#' Consistent colors for household classification throughout.
#'
#' @return Named vector of colors
#' @export
classification_colors <- function() {
  c(
    "stayer" = "#0072B2",  # Blue
    "coper" = "#E69F00",   # Orange
    "none" = "#999999"     # Grey
  )
}

#' Action colors for ENTER/EXIT/NO_CHANGE
#'
#' Consistent colors for action visualization.
#'
#' @return Named vector of colors
#' @export
action_colors <- function() {
  c(
    "ENTER_ENTERPRISE" = "#009E73",  # Green
    "EXIT_ENTERPRISE" = "#CC79A7",   # Pink
    "NO_CHANGE" = "#999999"          # Grey
  )
}

#' Transition colors for ENTER/EXIT/STAY
#'
#' @return Named vector of colors
#' @export
transition_colors <- function() {
  c(
    "ENTER" = "#009E73",  # Green
    "EXIT" = "#CC79A7",   # Pink
    "STAY" = "#999999"    # Grey
  )
}

# Set default theme for all subsequent plots
theme_set(theme_abm_minimal())
