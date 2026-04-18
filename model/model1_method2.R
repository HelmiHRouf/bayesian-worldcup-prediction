# =============================================================================
# fit_vi_model1.R — Fit Model 1: Hierarchical Team Strength Model
# =============================================================================

library(cmdstanr)
library(posterior)
library(bayesplot)
library(loo)
library(tidyverse)

# ---- 1. Load processed data ------------------------------------------------

stan_data <- readRDS("data/processed/stan_data.rds")

# Expected fields:
# N, T, E
# home_team, away_team
# era (match-level)
# home_goals, away_goals

# ---- 2. Fix naming ----------------------------------------------------------

#stan_data$home_score <- stan_data$home_goals
#stan_data$away_score <- stan_data$away_goals

# ---- 3. Construct team_era --------------------------------------------------

team_era_df <- tibble(
  team = c(stan_data$home_team, stan_data$away_team),
  era  = c(stan_data$era,       stan_data$era)
)

# Option A: most frequent era per team (recommended)
team_era <- team_era_df %>%
  count(team, era) %>%
  group_by(team) %>%
  slice_max(n, with_ties = FALSE) %>%
  arrange(team) %>%
  pull(era)

# Sanity check
if(length(team_era) != stan_data$T){
  stop("Mismatch: team_era length != T")
}

stan_data$team_era <- team_era

# ---- 4. Keep only required variables ----------------------------------------

stan_data <- list(
  N = stan_data$N,
  T = stan_data$T,
  E = stan_data$E,
  home_team = stan_data$home_team,
  away_team = stan_data$away_team,
  team_era = stan_data$team_era,
  home_score = stan_data$home_score,
  away_score = stan_data$away_score
)

# ---- 5. Compile model -------------------------------------------------------

model <- cmdstan_model("model/model1.stan")

# ---- 6. Variational Inference ----------------------------------------------

fit_vi <- model$variational(
  data = stan_data,
  seed = 2026,
  iter = 20000,            # number of VI iterations
  grad_samples = 1,
  elbo_samples = 100,
  output_samples = 2000    # draws from variational posterior
)

# ---- 7. Diagnostics ----------------------------------------------------------

fit_vi$cmdstan_diagnose()

fit_vi$summary(
  variables = c("mu", "sigma_att", "sigma_def")
) %>% print(n = 20)

# ---- 8. Extract draws --------------------------------------------------------

draws <- fit_vi$draws(format = "df")

# ---- 9. Era labels (REMOVED) -------------------------------------------------
# No external file needed

# ---- 10. Plot sigma_att ------------------------------------------------------

sigma_att_draws <- draws %>%
  select(starts_with("sigma_att")) %>%
  pivot_longer(everything(), names_to = "param", values_to = "value") %>%
  mutate(era = as.integer(str_extract(param, "\\d+")))

sigma_att_summary <- sigma_att_draws %>%
  group_by(era) %>%
  summarise(
    median = median(value),
    lo80   = quantile(value, 0.10),
    hi80   = quantile(value, 0.90),
    lo95   = quantile(value, 0.025),
    hi95   = quantile(value, 0.975),
    .groups = "drop"
  )

p_att <- ggplot(sigma_att_summary, aes(x = era, y = median)) +
  geom_ribbon(aes(ymin = lo95, ymax = hi95), alpha = 0.15) +
  geom_ribbon(aes(ymin = lo80, ymax = hi80), alpha = 0.30) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  scale_x_continuous(breaks = sigma_att_summary$era) +
  labs(
    title = "Attack Strength Dispersion Over Eras",
    subtitle = "Lower values → more parity → more unpredictability",
    x = "Era",
    y = expression(sigma[alpha,e])
  ) +
  theme_minimal(base_size = 14)

print(p_att)

# ---- 11. Plot sigma_def ------------------------------------------------------

sigma_def_draws <- draws %>%
  select(starts_with("sigma_def")) %>%
  pivot_longer(everything(), names_to = "param", values_to = "value") %>%
  mutate(era = as.integer(str_extract(param, "\\d+")))

sigma_def_summary <- sigma_def_draws %>%
  group_by(era) %>%
  summarise(
    median = median(value),
    lo80   = quantile(value, 0.10),
    hi80   = quantile(value, 0.90),
    lo95   = quantile(value, 0.025),
    hi95   = quantile(value, 0.975),
    .groups = "drop"
  )

p_def <- ggplot(sigma_def_summary, aes(x = era, y = median)) +
  geom_ribbon(aes(ymin = lo95, ymax = hi95), alpha = 0.15) +
  geom_ribbon(aes(ymin = lo80, ymax = hi80), alpha = 0.30) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  scale_x_continuous(breaks = sigma_def_summary$era) +
  labs(
    title = "Defense Strength Dispersion Over Eras",
    subtitle = "Lower values → more parity",
    x = "Era",
    y = expression(sigma[beta,e])
  ) +
  theme_minimal(base_size = 14)

print(p_def)
