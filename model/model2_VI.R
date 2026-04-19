# =============================================================================
# fit_model2VI.R — Fit Model 2: Continuous-Time Varying Competitive Balance
# =============================================================================

library(cmdstanr)
library(posterior)
library(bayesplot)
library(loo)
library(tidyverse)

# ---- 1. Load processed data ------------------------------------------------

matches <- read_csv("data/processed/matches.csv", show_col_types = FALSE)
team_mapping <- read_csv("data/processed/team_mapping.csv", show_col_types = FALSE)

# ---- 2. Prepare year indices ------------------------------------------------

# Get unique years and create year index mapping
year_mapping <- matches %>%
  distinct(year) %>%
  arrange(year) %>%
  mutate(year_id = row_number())

# Join year_id back to matches
matches <- matches %>%
  left_join(year_mapping, by = "year")

# ---- 3. Create Stan data list ---------------------------------------------

stan_data <- list(
  N = nrow(matches),
  T = nrow(team_mapping),
  Y = nrow(year_mapping),
  home_team = matches$home_team_id,
  away_team = matches$away_team_id,
  year = matches$year_id,
  home_score = matches$home_score,
  away_score = matches$away_score
)

# ---- 4. Save year mapping for later reference -----------------------------

write_csv(year_mapping, "data/processed/year_mapping.csv")

# ---- 5. Compile model -------------------------------------------------------

model <- cmdstan_model("model/model2.stan")

# ---- 6. Variational Inference ----------------------------------------------

fit_vi <- model$variational(
  data = stan_data,
  seed = 2026,
  iter = 20000,
  algorithm = "meanfield",  # ← fullrank is too slow for our model
  eta = 0.1,               # ← smaller step size for stability
  tol_rel_obj = 0.001,     # ← looser convergence tolerance
  draws = 2000
)

# Check if VI succeeded before continuing
if (fit_vi$return_codes() != 0) {
  warning("VI fitting may have issues. Check fit_vi$output() for details.")
}

# ---- 7. Diagnostics ----------------------------------------------------------

fit_vi$output()

print(fit_vi$summary(variables = c("mu", "omega", "rho", "sigma[1]", "sigma[20]", "sigma[27]")))

# ---- 8. Save fit_vi ------------------------------------------------------------

fit_vi$save_object("data/processed/vi_fit_model2.rds")

# ---- 9. Extract draws --------------------------------------------------------

draws <- fit_vi$draws(format = "df")

# ---- 10. Plot sigma trajectory over time ------------------------------------

sigma_draws <- draws %>%
  select(starts_with("sigma[")) %>%
  pivot_longer(everything(), names_to = "param", values_to = "value") %>%
  mutate(year_id = as.integer(str_extract(param, "\\d+")))

# Join with actual years
sigma_draws <- sigma_draws %>%
  left_join(year_mapping, by = "year_id")

sigma_summary <- sigma_draws %>%
  group_by(year, year_id) %>%
  summarise(
    median = median(value),
    lo80   = quantile(value, 0.10),
    hi80   = quantile(value, 0.90),
    lo95   = quantile(value, 0.025),
    hi95   = quantile(value, 0.975),
    .groups = "drop"
  )

p_sigma <- ggplot(sigma_summary, aes(x = year, y = median)) +
  geom_ribbon(aes(ymin = lo95, ymax = hi95), alpha = 0.15) +
  geom_ribbon(aes(ymin = lo80, ymax = hi80), alpha = 0.30) +
  geom_line(linewidth = 1, color = "steelblue") +
  geom_point(size = 2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  labs(
    title = "Evolution of Competitive Balance Over Time",
    subtitle = "Model 2: Time-Varying Team Strength Dispersion (sigma)",
    caption = "Lower sigma → teams more similar → more unpredictable",
    x = "Year",
    y = expression(sigma[y])
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    plot.caption = element_text(hjust = 0, face = "italic", color = "gray40")
  )

print(p_sigma)

# Save plot
ggsave("figures/model2VI_sigma_trajectory.png", p_sigma, width = 10, height = 6, dpi = 150)

# ---- 11. Plot log_sigma trajectory (random walk) ---------------------------

log_sigma_draws <- draws %>%
  select(starts_with("log_sigma[")) %>%
  pivot_longer(everything(), names_to = "param", values_to = "value") %>%
  mutate(year_id = as.integer(str_extract(param, "\\d+"))) %>%
  left_join(year_mapping, by = "year_id")

log_sigma_summary <- log_sigma_draws %>%
  group_by(year, year_id) %>%
  summarise(
    median = median(value),
    lo80   = quantile(value, 0.10),
    hi80   = quantile(value, 0.90),
    lo95   = quantile(value, 0.025),
    hi95   = quantile(value, 0.975),
    .groups = "drop"
  )

p_log_sigma <- ggplot(log_sigma_summary, aes(x = year, y = median)) +
  geom_ribbon(aes(ymin = lo95, ymax = hi95), alpha = 0.15) +
  geom_ribbon(aes(ymin = lo80, ymax = hi80), alpha = 0.30) +
  geom_line(linewidth = 1, color = "darkgreen") +
  geom_point(size = 2) +
  labs(
    title = "Random Walk on Log-Dispersion",
    subtitle = expression(paste("log ", sigma[y], " evolution with 80% and 95% credible intervals")),
    x = "Year",
    y = expression(log(sigma[y]))
  ) +
  theme_minimal(base_size = 14)

print(p_log_sigma)

ggsave("figures/model2VI_log_sigma_trajectory.png", p_log_sigma, width = 10, height = 6, dpi = 150)

# ---- 12. Compare with Model 1 (if available) -------------------------------

# Optional: Load Model 1 results and compare sigma estimates
if (file.exists("data/processed/vi_fit_era_dispersions.rds")) {

  message("\n=== Comparison with Model 1 ===")

  # Model 1 sigma by era (manually map to years for comparison)
  era_year_mapping <- tribble(
    ~era, ~year,
    1, 2000,   # 1998-2002 midpoint
    2, 2004,   # 2002-2006 midpoint
    3, 2008,   # 2006-2010 midpoint
    4, 2012,   # 2010-2014 midpoint
    5, 2016,   # 2014-2018 midpoint
    6, 2020,   # 2018-2022 midpoint
    7, 2024    # 2022-2026 midpoint
  )

  message("Model 1: Era-based sigma (discrete)")
  message("Model 2: Year-based sigma (continuous with random walk)")
}



message("\nModel 2 fitting complete!")
message("Results saved to:")
message("  - data/processed/vi_fit_model2.rds")
message("  - data/processed/loo_model2.rds")
message("  - figures/model2VI_sigma_trajectory.png")
message("  - figures/model2VI_log_sigma_trajectory.png")
