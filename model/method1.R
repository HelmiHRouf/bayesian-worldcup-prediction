# =============================================================================
# fit_model1.R — Fit Model 1: Hierarchical Team Strength Model
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

# ---- 6. Sample ---------------------------------------------------------------

fit <- model$sample(
  data            = stan_data,
  seed            = 2026,
  chains          = 4,
  parallel_chains = 4,
  iter_warmup     = 1000,
  iter_sampling   = 2000,
  refresh         = 500
)

# ---- 7. Diagnostics ----------------------------------------------------------

fit$cmdstan_diagnose()

fit$summary(
  variables = c("mu", "home_adv", "sigma_att", "sigma_def")
) %>% print(n = 20)

# ---- 8. Extract draws --------------------------------------------------------

draws <- fit$draws(format = "df")

# ---- 9. Era labels -----------------------------------------------------------

era_labels <- readRDS("data/processed/era_labels.rds")

cat("\nEra mapping:\n")
print(era_labels)

# ---- 10. Plot sigma_att ------------------------------------------------------

sigma_att_draws <- draws %>%
  select(starts_with("sigma_att")) %>%
  pivot_longer(everything(), names_to = "param", values_to = "value") %>%
  mutate(era = as.integer(str_extract(param, "\\d+"))) %>%
  left_join(era_labels, by = "era")

sigma_att_summary <- sigma_att_draws %>%
  group_by(era, era_label) %>%
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
  scale_x_continuous(
    breaks = era_labels$era,
    labels = era_labels$era_label
  ) +
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
  mutate(era = as.integer(str_extract(param, "\\d+"))) %>%
  left_join(era_labels, by = "era")

sigma_def_summary <- sigma_def_draws %>%
  group_by(era, era_label) %>%
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
  scale_x_continuous(
    breaks = era_labels$era,
    labels = era_labels$era_label
  ) +
  labs(
    title = "Defense Strength Dispersion Over Eras",
    subtitle = "Lower values → more parity",
    x = "Era",
    y = expression(sigma[beta,e])
  ) +
  theme_minimal(base_size = 14)

print(p_def)

# ---- 12. Posterior predictive check ------------------------------------------

home_rep <- fit$draws("home_score_rep", format = "matrix")
away_rep <- fit$draws("away_score_rep", format = "matrix")

p_ppc_home <- ppc_dens_overlay(stan_data$home_score, home_rep[1:200, ]) +
  theme_minimal()

p_ppc_away <- ppc_dens_overlay(stan_data$away_score, away_rep[1:200, ]) +
  theme_minimal()

print(p_ppc_home)
print(p_ppc_away)

# ---- 13. LOO-CV --------------------------------------------------------------

log_lik <- fit$draws("log_lik", format = "matrix")
loo_result <- loo(log_lik)
print(loo_result)

# ---- 14. Trace plots ---------------------------------------------------------

mcmc_trace(
  draws,
  pars = c("mu", "home_adv",
           paste0("sigma_att[", 1:stan_data$E, "]"))
) + theme_minimal()

cat("\nDone.\n")
