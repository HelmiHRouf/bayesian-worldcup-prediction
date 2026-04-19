################################################################################
# Model 2: Continuous-Time Varying Competitive Balance
# Implemented in simPPLe (importance-sampling PPL from STAT 405)
#
# Key idea of simPPLe:
#   - simulate() draws a latent variable from its prior
#   - observe()  multiplies the global `weight` by the likelihood of observed data
#   - posterior() runs many iterations and returns the weighted average of g(theta)
#
# IMPORTANT NOTE ON SCALE:
#   Pure importance sampling (what simPPLe does) degrades in high dimensions.
#   This implementation uses a SMALL dataset (few teams, few years) to keep
#   the effective sample size reasonable.  For the full World Cup dataset
#   you should use Stan (Model 2 Stan code) instead.
################################################################################

# ── 0. Load simPPLe ──────────────────────────────────────────────────────────
# Paste the simPPLe scaffold here directly so this file is self-contained.

suppressPackageStartupMessages(library(distr))

## Utilities
p_density <- function(distribution, realization) {
  d(distribution)(realization)
}

Bern <- function(prob) {
  DiscreteDistribution(supp = 0:1, prob = c(1 - prob, prob))
}

simulate <- function(distribution) {
  r(distribution)(1)
}

observe <- function(realization, distribution) {
  weight <<- weight * p_density(distribution, realization)
}

# simPPLe's posterior() – computes E[g(theta) | data] via importance sampling
posterior <- function(ppl_function, number_of_iterations) {
  numerator   <- 0.0
  denominator <- 0.0
  for (i in seq_len(number_of_iterations)) {
    weight <<- 1.0
    g_i         <- ppl_function()
    numerator   <- numerator   + weight * g_i
    denominator <- denominator + weight
  }
  return(numerator / denominator)
}

# Extended version: returns *all* weighted samples (needed for posterior plots)
posterior_samples <- function(ppl_function, number_of_iterations) {
  samples <- numeric(number_of_iterations)
  weights <- numeric(number_of_iterations)
  for (i in seq_len(number_of_iterations)) {
    weight    <<- 1.0
    samples[i] <- ppl_function()
    weights[i] <- weight
  }
  list(samples = samples, weights = weights)
}

# ── 1. Toy dataset ───────────────────────────────────────────────────────────
# Small example: 4 teams, 3 years, ~30 matches.
# Replace this with your real data slice (e.g., only World Cup matches,
# a handful of strong teams, 3-5 year windows).

set.seed(42)

teams <- c("Brazil", "Germany", "Argentina", "France")
T_teams <- length(teams)
years   <- c(2010, 2014, 2018)   # 3 year buckets
Y_years <- length(years)

# Each row: home_idx, away_idx, year_idx, home_goals, away_goals
# (indices are 1-based)
match_data <- data.frame(
  home_idx   = c(1, 2, 3, 4, 1, 2, 3, 4, 1, 3),
  away_idx   = c(2, 3, 4, 1, 3, 4, 1, 2, 4, 2),
  year_idx   = c(1, 1, 1, 1, 2, 2, 2, 2, 3, 3),
  home_goals = c(2, 1, 3, 1, 2, 0, 1, 2, 1, 2),
  away_goals = c(1, 1, 1, 2, 0, 1, 2, 1, 0, 1)
)
N_matches <- nrow(match_data)

# ── 2. Model 2 as a simPPLe function ─────────────────────────────────────────
#
# Generative story (mirrors the Stan model exactly):
#
#   mu             ~ Normal(0, 1)              [baseline log-rate]
#   log_sigma[1]   ~ Normal(0.5, 1)            [initial dispersion]
#   omega          ~ Exponential(rate=2)        [volatility of sigma]
#   log_sigma[y]   = log_sigma[y-1] + eps[y],  eps[y] ~ Normal(0, omega^2)
#   sigma[y]       = exp(log_sigma[y])
#   attack[t,y]    ~ Normal(0, sigma[y]^2)
#   defense[t,y]   ~ Normal(0, sigma[y]^2)
#   home_goals[i]  ~ Poisson(exp(mu + attack[h,y] - defense[a,y]))
#   away_goals[i]  ~ Poisson(exp(mu + attack[a,y] - defense[h,y]))
#
# The function returns sigma[y] for the year of interest (here all years
# via a list, but posterior() needs a scalar — see wrappers below).

model2 <- function(return_what = "sigma_all") {

  # --- Global baseline ---
  mu <- simulate(Norm(mean = 0, sd = 1))

  # --- Time-varying log-sigma (random walk) ---
  log_sigma <- numeric(Y_years)
  log_sigma[1] <- simulate(Norm(mean = 0.5, sd = 1))

  omega <- simulate(Exp(rate = 2))   # distr uses rate parameterisation

  for (y in seq(2, Y_years)) {
    eps           <- simulate(Norm(mean = 0, sd = omega))
    log_sigma[y]  <- log_sigma[y - 1] + eps
  }
  sigma <- exp(log_sigma)

  # --- Team-year attack and defense strengths ---
  attack  <- matrix(0, nrow = T_teams, ncol = Y_years)
  defense <- matrix(0, nrow = T_teams, ncol = Y_years)

  for (y in seq_len(Y_years)) {
    for (t in seq_len(T_teams)) {
      attack[t, y]  <- simulate(Norm(mean = 0, sd = sigma[y]))
      defense[t, y] <- simulate(Norm(mean = 0, sd = sigma[y]))
    }
  }

  # --- Likelihood: observe all match scores ---
  for (i in seq_len(N_matches)) {
    h <- match_data$home_idx[i]
    a <- match_data$away_idx[i]
    y <- match_data$year_idx[i]

    log_lambda_home <- mu + attack[h, y] - defense[a, y]
    log_lambda_away <- mu + attack[a, y] - defense[h, y]

    # Clamp log-rates to avoid numerical explosions (Poisson support is [0, Inf))
    log_lambda_home <- max(min(log_lambda_home, 5), -5)
    log_lambda_away <- max(min(log_lambda_away, 5), -5)

    observe(match_data$home_goals[i], Pois(lambda = exp(log_lambda_home)))
    observe(match_data$away_goals[i], Pois(lambda = exp(log_lambda_away)))
  }

  # --- Return quantity of interest ---
  if (return_what == "sigma_all") {
    return(sigma)          # vector – use posterior_samples() for this
  } else if (return_what == "sigma1") {
    return(sigma[1])
  } else if (return_what == "sigma2") {
    return(sigma[2])
  } else if (return_what == "sigma3") {
    return(sigma[3])
  } else if (return_what == "mu") {
    return(mu)
  } else if (return_what == "omega") {
    return(omega)
  }
}

# ── 3. Scalar wrappers for posterior() ───────────────────────────────────────
model2_sigma1 <- function() model2("sigma1")
model2_sigma2 <- function() model2("sigma2")
model2_sigma3 <- function() model2("sigma3")
model2_mu     <- function() model2("mu")
model2_omega  <- function() model2("omega")

# ── 4. Run inference ──────────────────────────────────────────────────────────
cat("Running simPPLe importance sampling...\n")
cat("(This may take a minute — importance sampling scales poorly in high dim)\n\n")

set.seed(123)
n_iter <- 5000   # increase for better estimates; decrease if too slow

# Posterior means
E_sigma1 <- posterior(model2_sigma1, n_iter)
E_sigma2 <- posterior(model2_sigma2, n_iter)
E_sigma3 <- posterior(model2_sigma3, n_iter)
E_mu     <- posterior(model2_mu,     n_iter)
E_omega  <- posterior(model2_omega,  n_iter)

cat("=== Posterior Means ===\n")
cat(sprintf("  mu      (baseline log-rate):        %.4f\n", E_mu))
cat(sprintf("  omega   (sigma volatility):          %.4f\n", E_omega))
cat(sprintf("  sigma[%d] (team strength spread, %d): %.4f\n", 1, years[1], E_sigma1))
cat(sprintf("  sigma[%d] (team strength spread, %d): %.4f\n", 2, years[2], E_sigma2))
cat(sprintf("  sigma[%d] (team strength spread, %d): %.4f\n", 3, years[3], E_sigma3))

cat("\n[Interpretation]\n")
cat("  Large sigma_y  => teams differ a lot  => less competitive (dominant teams)\n")
cat("  Small sigma_y  => teams are similar   => more competitive (parity)\n")

# ── 5. Posterior distribution plots via weighted samples ─────────────────────
set.seed(456)
n_plot <- 3000

cat("\nCollecting weighted samples for plotting...\n")
samp1 <- posterior_samples(model2_sigma1, n_plot)
samp2 <- posterior_samples(model2_sigma2, n_plot)
samp3 <- posterior_samples(model2_sigma3, n_plot)

# Normalise weights for plotting
norm_w1 <- samp1$weights / sum(samp1$weights)
norm_w2 <- samp2$weights / sum(samp2$weights)
norm_w3 <- samp3$weights / sum(samp3$weights)

# Effective sample size (ESS) — a sanity check for importance sampling quality
ess <- function(w) { 1 / sum((w / sum(w))^2) }
cat(sprintf("\nEffective Sample Sizes (out of %d):\n", n_plot))
cat(sprintf("  sigma[1]: ESS = %.1f\n", ess(samp1$weights)))
cat(sprintf("  sigma[2]: ESS = %.1f\n", ess(samp2$weights)))
cat(sprintf("  sigma[3]: ESS = %.1f\n", ess(samp3$weights)))
cat("  (ESS << n_iter means the model is too high-dimensional for simPPLe;\n")
cat("   consider Stan for the full dataset.)\n")

# Draw weighted samples (resample with replacement)
draw_weighted <- function(samp_obj, n_draw = 1000) {
  w <- samp_obj$weights / sum(samp_obj$weights)
  idx <- sample(seq_along(w), size = n_draw, replace = TRUE, prob = w)
  samp_obj$samples[idx]
}

draws1 <- draw_weighted(samp1)
draws2 <- draw_weighted(samp2)
draws3 <- draw_weighted(samp3)

# Plot posterior densities of sigma over time
png("sigma_posterior.png", width = 800, height = 500)
xlim_range <- range(c(draws1, draws2, draws3), na.rm = TRUE)
xlim_range <- c(0, min(xlim_range[2], 5))  # cap for readability

d1 <- density(draws1, from = 0, to = xlim_range[2])
d2 <- density(draws2, from = 0, to = xlim_range[2])
d3 <- density(draws3, from = 0, to = xlim_range[2])

ylim_max <- max(d1$y, d2$y, d3$y) * 1.1

plot(d1, col = "steelblue", lwd = 2, xlim = xlim_range, ylim = c(0, ylim_max),
     main = "Posterior of sigma_y: Team Strength Spread Over Time\n(simPPLe / Importance Sampling)",
     xlab = "sigma_y  (larger = less competitive)", ylab = "Density")
lines(d2, col = "firebrick", lwd = 2)
lines(d3, col = "darkgreen", lwd = 2)
legend("topright",
       legend = paste0("sigma[", 1:3, "] (", years, ")"),
       col    = c("steelblue", "firebrick", "darkgreen"),
       lwd    = 2, bty = "n")
abline(v = c(E_sigma1, E_sigma2, E_sigma3),
       col = c("steelblue", "firebrick", "darkgreen"),
       lty = 2, lwd = 1.5)
dev.off()

cat("\nPosterior density plot saved to: sigma_posterior.png\n")

# ── 6. Posterior predictive check (one match as example) ─────────────────────
cat("\n=== Posterior Predictive Check (match 1: Brazil vs Germany, 2010) ===\n")

set.seed(789)
model2_ppc <- function() {
  mu           <- simulate(Norm(mean = 0, sd = 1))
  log_sigma    <- numeric(Y_years)
  log_sigma[1] <- simulate(Norm(mean = 0.5, sd = 1))
  omega        <- simulate(Exp(rate = 2))
  for (y in seq(2, Y_years)) {
    log_sigma[y] <- log_sigma[y - 1] + simulate(Norm(mean = 0, sd = omega))
  }
  sigma <- exp(log_sigma)

  attack  <- matrix(0, T_teams, Y_years)
  defense <- matrix(0, T_teams, Y_years)
  for (y in seq_len(Y_years))
    for (t in seq_len(T_teams)) {
      attack[t, y]  <- simulate(Norm(0, sigma[y]))
      defense[t, y] <- simulate(Norm(0, sigma[y]))
    }

  # Observe ALL matches to condition the posterior
  for (i in seq_len(N_matches)) {
    h <- match_data$home_idx[i]; a <- match_data$away_idx[i]; y <- match_data$year_idx[i]
    llh <- max(min(mu + attack[h,y] - defense[a,y], 5), -5)
    lla <- max(min(mu + attack[a,y] - defense[h,y], 5), -5)
    observe(match_data$home_goals[i], Pois(exp(llh)))
    observe(match_data$away_goals[i], Pois(exp(lla)))
  }

  # Predict match 1: Brazil (1) vs Germany (2), year 1
  llh_pred <- max(min(mu + attack[1, 1] - defense[2, 1], 5), -5)
  lla_pred <- max(min(mu + attack[2, 1] - defense[1, 1], 5), -5)
  rpois(1, exp(llh_pred)) - rpois(1, exp(lla_pred))  # goal difference (+ = home win)
}

ppc_samp <- posterior_samples(model2_ppc, 2000)
goal_diff_draws <- draw_weighted(ppc_samp, 1000)

cat(sprintf("  Predicted goal difference (Brazil - Germany):\n"))
cat(sprintf("    Mean: %.2f\n", mean(goal_diff_draws)))
cat(sprintf("    P(home win):  %.2f\n", mean(goal_diff_draws > 0)))
cat(sprintf("    P(draw):      %.2f\n", mean(goal_diff_draws == 0)))
cat(sprintf("    P(away win):  %.2f\n", mean(goal_diff_draws < 0)))

cat("\n=== Done ===\n")
cat("To use with your real dataset:\n")
cat("  1. Replace `match_data` with your filtered real data (small slice first!)\n")
cat("  2. Adjust T_teams, Y_years, and years accordingly\n")
cat("  3. For the full dataset, use the Stan model — simPPLe ESS will be too low\n")
