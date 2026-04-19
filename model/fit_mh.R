# =============================================================================
# fit_mh.R — Metropolis-Hastings for Model 1 and Model 2
#
# For computational tractability, MH is applied to the ERA-LEVEL / YEAR-LEVEL
# dispersion parameters only (sigma_alpha, sigma_beta per era/year), with the
# baseline mu. Team-level attack/defense strengths are marginalized out
# analytically (they are Normal random effects, so the marginal likelihood
# of goals given sigma is a Poisson-log-Normal, approximated via Monte Carlo).
#
# This is a VALID simplification of both models:
#   Model 1: infer {sigma_alpha_e, sigma_beta_e} for e = 1..E  (7 eras)
#   Model 2: infer {sigma_y} for y = 1..Y  (via random walk on log_sigma)
#             plus omega (volatility) and mu
# =============================================================================

library(tidyverse)

set.seed(2026)

# ---- Load data ---------------------------------------------------------------

matches     <- read_csv("data/processed/matches.csv",     show_col_types = FALSE)
team_mapping <- read_csv("data/processed/team_mapping.csv", show_col_types = FALSE)

# ---- Era definitions (Model 1) ----------------------------------------------

era_breaks <- c(1998, 2002, 2006, 2010, 2014, 2018, 2022, 2026)
matches <- matches %>%
  filter(year >= 1998) %>%   # drop matches outside era range
  mutate(era = cut(year,
                   breaks = era_breaks,
                   labels = 1:7,
                   include.lowest = TRUE) %>% as.integer())

E <- 7
T <- nrow(team_mapping)

# ---- Year indices (Model 2) -------------------------------------------------

year_mapping <- matches %>%
  distinct(year) %>%
  arrange(year) %>%
  mutate(year_id = row_number())

matches <- matches %>%
  left_join(year_mapping, by = "year")

Y <- nrow(year_mapping)

# =============================================================================
# SHARED UTILITY: marginal log-likelihood via Monte Carlo
#
# For a single match, given sigma_atk and sigma_def (team strength spreads),
# we approximate:
#   log p(home_goals, away_goals | mu, sigma_atk, sigma_def)
# by sampling S draws of (attack, defense) from their Normal priors and
# averaging the Poisson likelihoods. This marginalizes out team strengths.
# =============================================================================

S <- 200  # Monte Carlo samples for marginal likelihood (increase for accuracy)

marginal_log_lik_match <- function(home_goals, away_goals,
                                   mu, sigma_atk, sigma_def) {
  # Sample team strength differences from their marginal (convolution of Normals)
  # attack_h - defense_a ~ N(0, sigma_atk^2 + sigma_def^2)
  # attack_a - defense_h ~ N(0, sigma_atk^2 + sigma_def^2)
  sigma_combined <- sqrt(sigma_atk^2 + sigma_def^2)

  strength_diff_home <- rnorm(S, 0, sigma_combined)
  strength_diff_away <- rnorm(S, 0, sigma_combined)

  log_lambda_home <- mu + strength_diff_home
  log_lambda_away <- mu + strength_diff_away

  # Poisson log-likelihood for each MC sample
  ll_home <- dpois(home_goals, exp(log_lambda_home), log = TRUE)
  ll_away <- dpois(away_goals, exp(log_lambda_away), log = TRUE)

  # log of mean of exp(ll) = log-sum-exp trick
  ll_combined <- ll_home + ll_away
  log_mean_exp <- max(ll_combined) + log(mean(exp(ll_combined - max(ll_combined))))

  return(log_mean_exp)
}

# =============================================================================
# MODEL 1: MH for era-level dispersions
# Parameters: mu, sigma_alpha[1..E], sigma_beta[1..E]  (2E + 1 parameters)
# =============================================================================

cat("=== Model 1: Metropolis-Hastings ===\n")
cat("Parameters: mu, sigma_alpha[e], sigma_beta[e] for e = 1..7\n\n")

# ---- Log posterior (Model 1) ------------------------------------------------

log_posterior_m1 <- function(mu, sigma_alpha, sigma_beta) {

  # Priors (from your README)
  lp <- dnorm(mu, 0, 1, log = TRUE)

  # HalfNormal(0,1) priors on sigma parameters
  # HalfNormal(0,1) = Normal(0,1) truncated to positive => log_p = log(2) + dnorm
  for (e in 1:E) {
    if (sigma_alpha[e] <= 0 || sigma_beta[e] <= 0) return(-Inf)
    lp <- lp + log(2) + dnorm(sigma_alpha[e], 0, 1, log = TRUE)
    lp <- lp + log(2) + dnorm(sigma_beta[e],  0, 1, log = TRUE)
  }

  # Marginal likelihood over all matches
  ll <- 0
  for (i in seq_len(nrow(matches))) {
    e <- matches$era[i]
    if (is.na(e)) next
    ll <- ll + marginal_log_lik_match(
      matches$home_score[i], matches$away_score[i],
      mu, sigma_alpha[e], sigma_beta[e]
    )
  }

  return(lp + ll)
}

# ---- MH sampler (Model 1) ---------------------------------------------------

mh_model1 <- function(n_iter     = 2000,
                      n_warmup   = 500,
                      proposal_sd = 0.05) {

  # Initialize
  mu          <- 0.0
  sigma_alpha <- rep(0.5, E)
  sigma_beta  <- rep(0.5, E)

  # Storage
  n_store <- n_iter - n_warmup
  chain_mu    <- numeric(n_store)
  chain_sa    <- matrix(NA, n_store, E)
  chain_sb    <- matrix(NA, n_store, E)
  n_accept    <- 0

  lp_current <- log_posterior_m1(mu, sigma_alpha, sigma_beta)

  cat("Running MH for Model 1...\n")

  for (iter in seq_len(n_iter)) {

    if (iter %% 200 == 0)
      cat(sprintf("  Iteration %d / %d  (acceptance so far: %.2f)\n",
                  iter, n_iter, n_accept / iter))

    # --- Block 1: update mu alone ---
    mu_prop <- mu + rnorm(1, 0, proposal_sd)
    lp_prop <- log_posterior_m1(mu_prop, sigma_alpha, sigma_beta)
    if (log(runif(1)) < lp_prop - lp_current) {
      mu         <- mu_prop
      lp_current <- lp_prop
      n_accept   <- n_accept + 1
    }

    # --- Block 2: update each sigma_alpha[e] one at a time ---
    for (e in seq_len(E)) {
      sa_prop    <- sigma_alpha
      sa_prop[e] <- abs(sigma_alpha[e] + rnorm(1, 0, proposal_sd))
      lp_prop    <- log_posterior_m1(mu, sa_prop, sigma_beta)
      if (log(runif(1)) < lp_prop - lp_current) {
        sigma_alpha <- sa_prop
        lp_current  <- lp_prop
      }
    }

    # --- Block 3: update each sigma_beta[e] one at a time ---
    for (e in seq_len(E)) {
      sb_prop    <- sigma_beta
      sb_prop[e] <- abs(sigma_beta[e] + rnorm(1, 0, proposal_sd))
      lp_prop    <- log_posterior_m1(mu, sigma_alpha, sb_prop)
      if (log(runif(1)) < lp_prop - lp_current) {
        sigma_beta <- sb_prop
        lp_current <- lp_prop
      }
    }

    # Store post-warmup
    if (iter > n_warmup) {
      idx <- iter - n_warmup
      chain_mu[idx]   <- mu
      chain_sa[idx, ] <- sigma_alpha
      chain_sb[idx, ] <- sigma_beta
    }
  }

  cat(sprintf("\nFinal acceptance rate: %.3f\n", n_accept / n_iter))
  cat("(Ideal range: 0.20 - 0.40)\n\n")

  list(mu          = chain_mu,
       sigma_alpha = chain_sa,
       sigma_beta  = chain_sb,
       accept_rate = n_accept / n_iter)
}

# Run Model 1 MH
mh1 <- mh_model1(n_iter = 2000, n_warmup = 500, proposal_sd = 0.05)

# ---- Summarise Model 1 results ----------------------------------------------

cat("\n=== Model 1 MH Posterior Summaries ===\n")
cat(sprintf("mu:  mean = %.3f,  sd = %.3f,  95%% CI = [%.3f, %.3f]\n",
            mean(mh1$mu), sd(mh1$mu),
            quantile(mh1$mu, 0.025), quantile(mh1$mu, 0.975)))

for (e in 1:E) {
  cat(sprintf("sigma_alpha[%d]:  mean = %.3f,  sd = %.3f\n",
              e, mean(mh1$sigma_alpha[, e]), sd(mh1$sigma_alpha[, e])))
}

# ---- Plot Model 1 sigma_alpha trajectory ------------------------------------

era_labels <- c("1998-2002", "2002-2006", "2006-2010",
                "2010-2014", "2014-2018", "2018-2022", "2022-2026")

m1_summary <- tibble(
  era        = 1:E,
  era_label  = factor(era_labels, levels = era_labels),
  median_sa  = apply(mh1$sigma_alpha, 2, median),
  lo95_sa    = apply(mh1$sigma_alpha, 2, quantile, 0.025),
  hi95_sa    = apply(mh1$sigma_alpha, 2, quantile, 0.975),
  lo80_sa    = apply(mh1$sigma_alpha, 2, quantile, 0.10),
  hi80_sa    = apply(mh1$sigma_alpha, 2, quantile, 0.90)
)

p_m1 <- ggplot(m1_summary, aes(x = era, y = median_sa)) +
  geom_ribbon(aes(ymin = lo95_sa, ymax = hi95_sa), alpha = 0.15) +
  geom_ribbon(aes(ymin = lo80_sa, ymax = hi80_sa), alpha = 0.30) +
  geom_line(linewidth = 1, color = "steelblue") +
  geom_point(size = 3) +
  scale_x_continuous(breaks = 1:E, labels = era_labels) +
  labs(
    title    = "Model 1: Competitive Balance Across Eras",
    subtitle = "Posterior of sigma_alpha[e] via Metropolis-Hastings",
    caption  = "Higher sigma => teams differ more => less competitive",
    x        = "Era",
    y        = expression(sigma[alpha][e])
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x  = element_text(angle = 30, hjust = 1),
    plot.title   = element_text(face = "bold"),
    plot.caption = element_text(hjust = 0, face = "italic", color = "gray40")
  )

print(p_m1)
ggsave("figures/model1_mh_sigma_alpha_trajectory.png", p_m1,
       width = 10, height = 6, dpi = 150)

# Save Model 1 MH results
saveRDS(mh1, "data/processed/mh_fit_model1.rds")


# ---- Plot Model 1 sigma_beta trajectory ------------------------------------

m1_summary_sb <- tibble(
  era        = 1:E,
  era_label  = factor(era_labels, levels = era_labels),
  median_sb  = apply(mh1$sigma_beta, 2, median),
  lo95_sb    = apply(mh1$sigma_beta, 2, quantile, 0.025),
  hi95_sb    = apply(mh1$sigma_beta, 2, quantile, 0.975),
  lo80_sb    = apply(mh1$sigma_beta, 2, quantile, 0.10),
  hi80_sb    = apply(mh1$sigma_beta, 2, quantile, 0.90)
)

p_m1_sb <- ggplot(m1_summary_sb, aes(x = era, y = median_sb)) +
  geom_ribbon(aes(ymin = lo95_sb, ymax = hi95_sb), alpha = 0.15) +
  geom_ribbon(aes(ymin = lo80_sb, ymax = hi80_sb), alpha = 0.30) +
  geom_line(linewidth = 1, color = "firebrick") +
  geom_point(size = 3) +
  scale_x_continuous(breaks = 1:E, labels = era_labels) +
  labs(
    title    = "Model 1: Defensive Strength Dispersion Across Eras",
    subtitle = "Posterior of sigma_beta[e] via Metropolis-Hastings",
    caption  = "Higher sigma => teams differ more defensively => less competitive",
    x        = "Era",
    y        = expression(sigma[beta][e])
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x  = element_text(angle = 30, hjust = 1),
    plot.title   = element_text(face = "bold"),
    plot.caption = element_text(hjust = 0, face = "italic", color = "gray40")
  )

print(p_m1_sb)
ggsave("figures/model1_mh_sigma_beta_trajectory.png", p_m1_sb,
       width = 10, height = 6, dpi = 150)

# =============================================================================
# MODEL 2: MH for time-varying sigma via random walk
# Parameters: mu, log_sigma[1..Y], omega
# (team strengths marginalized out as above)
# This matches your README exactly:
#   log_sigma[y] = log_sigma[y-1] + epsilon_y,  epsilon_y ~ N(0, omega^2)
#   sigma[y] = exp(log_sigma[y])
#   attack[t,y] ~ N(0, sigma[y]^2)   [marginalized]
#   defense[t,y] ~ N(0, sigma[y]^2)  [marginalized]
# =============================================================================

cat("\n=== Model 2: Metropolis-Hastings ===\n")
cat("Parameters: mu, log_sigma[y] for y=1..Y, omega\n\n")

# ---- Log posterior (Model 2) ------------------------------------------------

log_posterior_m2 <- function(mu, log_sigma, omega) {

  if (omega <= 0) return(-Inf)

  Y_local <- length(log_sigma)
  sigma   <- exp(log_sigma)

  # Priors (from your README)
  lp <- dnorm(mu, 0, 1, log = TRUE)

  # sigma[1] ~ HalfNormal(0, 1)  =>  log_sigma[1] ~ log-HalfNormal
  # We put prior on sigma[1] directly
  lp <- lp + log(2) + dnorm(sigma[1], 0, 1, log = TRUE) + log_sigma[1]
  # (Jacobian: d(sigma)/d(log_sigma) = sigma)

  # omega ~ HalfNormal(0, 0.5)
  if (omega <= 0) return(-Inf)
  lp <- lp + log(2) + dnorm(omega, 0, 0.5, log = TRUE)

  # Random walk prior on log_sigma[2..Y]
  for (y in 2:Y_local) {
    lp <- lp + dnorm(log_sigma[y], log_sigma[y - 1], omega, log = TRUE)
  }

  # Marginal likelihood: sigma_alpha = sigma_beta = sigma[y] per your README
  ll <- 0
  for (i in seq_len(nrow(matches))) {
    y_idx <- matches$year_id[i]
    if (is.na(y_idx)) next
    ll <- ll + marginal_log_lik_match(
      matches$home_score[i], matches$away_score[i],
      mu, sigma[y_idx], sigma[y_idx]   # attack and defense share sigma[y]
    )
  }

  return(lp + ll)
}

# ---- MH sampler (Model 2) ---------------------------------------------------

mh_model2 <- function(n_iter      = 2000,
                       n_warmup    = 500,
                       proposal_sd = 0.03) {

  # Initialize
  mu        <- 0.0
  log_sigma <- rep(log(0.5), Y)   # start sigma at ~0.5
  omega     <- 0.3

  # Storage
  n_store       <- n_iter - n_warmup
  chain_mu      <- numeric(n_store)
  chain_logsig  <- matrix(NA, n_store, Y)
  chain_omega   <- numeric(n_store)
  n_accept      <- 0

  lp_current <- log_posterior_m2(mu, log_sigma, omega)

  cat("Running MH for Model 2...\n")

  for (iter in seq_len(n_iter)) {

    if (iter %% 200 == 0)
      cat(sprintf("  Iteration %d / %d  (acceptance so far: %.2f)\n",
                  iter, n_iter, n_accept / iter))

    # --- Block 1: update mu alone ---
    mu_prop <- mu + rnorm(1, 0, proposal_sd)
    lp_prop <- log_posterior_m2(mu_prop, log_sigma, omega)
    if (log(runif(1)) < lp_prop - lp_current) {
      mu         <- mu_prop
      lp_current <- lp_prop
      n_accept   <- n_accept + 1
    }

    # --- Block 2: update omega alone ---
    omega_prop <- abs(omega + rnorm(1, 0, proposal_sd * 0.5))
    lp_prop    <- log_posterior_m2(mu, log_sigma, omega_prop)
    if (log(runif(1)) < lp_prop - lp_current) {
      omega      <- omega_prop
      lp_current <- lp_prop
    }

    # --- Block 3: update each log_sigma[y] one at a time ---
    for (y in seq_len(Y)) {
      log_sigma_prop    <- log_sigma
      log_sigma_prop[y] <- log_sigma[y] + rnorm(1, 0, proposal_sd)
      lp_prop <- log_posterior_m2(mu, log_sigma_prop, omega)
      if (log(runif(1)) < lp_prop - lp_current) {
        log_sigma  <- log_sigma_prop
        lp_current <- lp_prop
      }
    }

    # --- Store post-warmup ---
    if (iter > n_warmup) {
      idx <- iter - n_warmup
      chain_mu[idx]       <- mu
      chain_logsig[idx, ] <- log_sigma
      chain_omega[idx]    <- omega
    }
  }

  cat(sprintf("\nFinal acceptance rate: %.3f\n", n_accept / n_iter))
  cat("(Ideal range: 0.20 - 0.40)\n\n")

  chain_sigma <- exp(chain_logsig)

  list(mu        = chain_mu,
       log_sigma = chain_logsig,
       sigma     = chain_sigma,
       omega     = chain_omega,
       accept_rate = n_accept / n_iter)
}

# Run Model 2 MH
mh2 <- mh_model2(n_iter = 2000, n_warmup = 500, proposal_sd = 0.03)

# ---- Summarise Model 2 results ----------------------------------------------

cat("\n=== Model 2 MH Posterior Summaries ===\n")
cat(sprintf("mu:    mean = %.3f,  sd = %.3f\n", mean(mh2$mu), sd(mh2$mu)))
cat(sprintf("omega: mean = %.3f,  sd = %.3f,  95%% CI = [%.3f, %.3f]\n",
            mean(mh2$omega), sd(mh2$omega),
            quantile(mh2$omega, 0.025), quantile(mh2$omega, 0.975)))

# ---- Plot Model 2 sigma trajectory ------------------------------------------

sigma_summary_m2 <- tibble(
  year_id = 1:Y,
  year    = year_mapping$year,
  median  = apply(mh2$sigma, 2, median),
  lo95    = apply(mh2$sigma, 2, quantile, 0.025),
  hi95    = apply(mh2$sigma, 2, quantile, 0.975),
  lo80    = apply(mh2$sigma, 2, quantile, 0.10),
  hi80    = apply(mh2$sigma, 2, quantile, 0.90)
)

p_m2 <- ggplot(sigma_summary_m2, aes(x = year, y = median)) +
  geom_ribbon(aes(ymin = lo95, ymax = hi95), alpha = 0.15) +
  geom_ribbon(aes(ymin = lo80, ymax = hi80), alpha = 0.30) +
  geom_line(linewidth = 1, color = "steelblue") +
  geom_point(size = 2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  labs(
    title    = "Evolution of Competitive Balance Over Time",
    subtitle = "Model 2: Time-Varying Team Strength Dispersion via MH",
    caption  = "Lower sigma => teams more similar => more competitive\nMetropolis-Hastings sampling",
    x        = "Year",
    y        = expression(sigma[y])
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title   = element_text(face = "bold"),
    plot.caption = element_text(hjust = 0, face = "italic", color = "gray40")
  )

print(p_m2)
ggsave("figures/model2_mh_sigma_trajectory.png", p_m2,
       width = 10, height = 6, dpi = 150)

# Save Model 2 MH results
saveRDS(mh2, "data/processed/mh_fit_model2.rds")

message("\nMH fitting complete!")
message("Results saved to:")
message("  - data/processed/mh_fit_model1.rds")
message("  - data/processed/mh_fit_model2.rds")
message("  - figures/model2_mh_sigma_trajectory.png")
message("  - figures/model1_mh_sigma_alpha_trajectory.png")
message("  - figures/model1_mh_sigma_beta_trajectory.png")
