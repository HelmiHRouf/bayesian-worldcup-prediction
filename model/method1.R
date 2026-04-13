# =============================================================================
# fit_model1.R — Fit Model 1: Shrinking Strength Spread (Poisson Goals)
# =============================================================================

library(cmdstanr)
library(posterior)
library(bayesplot)
library(loo)
library(tidyverse)

# ---- 1. Load processed data ------------------------------------------------

stan_data <- readRDS("data/processed/stan_data.rds")

# ---- 3. Compile model -------------------------------------------------------

model <- cmdstan_model("model/model1.stan")

# ---- 4. Sample ---------------------------------------------------------------

fit <- model$sample(
  data            = stan_data,
  seed            = 2026,
  chains          = 1,
  iter_warmup     = 1000,
  iter_sampling   = 5000,
  refresh         = 500
)

# ---- 5. Diagnostics ----------------------------------------------------------

fit$cmdstan_diagnose()
fit$summary(variables = c("mu", "delta", "sigma")) %>% print(n = 20)

# ---- 6. Extract draws --------------------------------------------------------

draws <- fit$draws(format = "df")

# Era labels — adjust if your data uses different era definitions
era_labels <- train %>%
  distinct(era, era_label) %>%
  arrange(era)

cat("\nEra mapping:\n")
print(era_labels)

# ---- 7. Plot: sigma trend across eras (key result) --------------------------

sigma_draws <- draws %>%
  select(starts_with("sigma")) %>%
  pivot_longer(everything(), names_to = "param", values_to = "value") %>%
  mutate(era = as.integer(str_extract(param, "\\d+"))) %>%
  left_join(era_labels, by = "era")

# Posterior intervals
sigma_summary <- sigma_draws %>%
  group_by(era, era_label) %>%
  summarise(
    median = median(value),
    lo80   = quantile(value, 0.10),
    hi80   = quantile(value, 0.90),
    lo95   = quantile(value, 0.025),
    hi95   = quantile(value, 0.975),
    .groups = "drop"
  )

p_trend <- ggplot(sigma_summary, aes(x = era, y = median)) +
  geom_ribbon(aes(ymin = lo95, ymax = hi95), alpha = 0.15, fill = "steelblue") +
  geom_ribbon(aes(ymin = lo80, ymax = hi80), alpha = 0.30, fill = "steelblue") +
  geom_line(linewidth = 1, color = "steelblue") +
  geom_point(size = 3, color = "steelblue") +
  scale_x_continuous(breaks = era_labels$era, labels = era_labels$era_label) +
  labs(
    title    = "Team Strength Spread Over Eras (Model 1)",
    subtitle = "Decreasing sigma → more competitive balance (parity)",
    x = "Era", y = expression(sigma[e])
  ) +
  theme_minimal(base_size = 14)

print(p_trend)
ggsave("figures/model1_sigma_trend.png", p_trend, width = 8, height = 5, dpi = 300)

# ---- 8. Posterior predictive check -------------------------------------------

home_rep <- fit$draws("home_goals_rep", format = "matrix")
away_rep <- fit$draws("away_goals_rep", format = "matrix")

p_ppc_home <- ppc_dens_overlay(stan_data$home_goals, home_rep[1:200, ]) +
  labs(title = "PPC: Home Goals") +
  theme_minimal()

p_ppc_away <- ppc_dens_overlay(stan_data$away_goals, away_rep[1:200, ]) +
  labs(title = "PPC: Away Goals") +
  theme_minimal()

print(p_ppc_home)
print(p_ppc_away)

ggsave("figures/model1_ppc_home.png", p_ppc_home, width = 7, height = 4, dpi = 300)
ggsave("figures/model1_ppc_away.png", p_ppc_away, width = 7, height = 4, dpi = 300)

# ---- 9. LOO-CV ---------------------------------------------------------------

log_lik <- fit$draws("log_lik", format = "matrix")
loo_result <- loo(log_lik)
print(loo_result)

# ---- 10. Trace plots for key parameters -------------------------------------

mcmc_trace(draws, pars = c("mu", "delta", paste0("sigma[", 1:stan_data$E, "]"))) +
  theme_minimal()

cat("\nDone. Figures saved to figures/\n")
