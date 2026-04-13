// Model 1: Shrinking Strength Spread (Poisson score)
//
// Team-level attack/defense effects are marginalized out via Negative Binomial.
// NegBin(mu, phi) ≈ Poisson-LogNormal where phi = 1/sigma^2.
// Key quantity of interest: sigma[e] = spread of team strengths per era.
//   - Large sigma  → teams differ a lot  → less competitive
//   - Small sigma  → teams are similar   → more competitive (parity)

data {
  int<lower=1> N;                         // number of matches
  int<lower=1> E;                         // number of eras
  array[N] int<lower=1, upper=E> era;     // era index per match
  array[N] int<lower=0> home_score;       // score scored by home team
  array[N] int<lower=0> away_score;       // score scored by away team
}

parameters {
  real mu;                    // global baseline log scoring rate
  vector<lower=0>[E] sigma;  // team strength spread per era
}

model {
  // --- Priors ---
  mu ~ normal(0, 1);
  sigma ~ normal(0, 1);       // half-normal via lower=0 constraint

  // --- Likelihood ---
  // phi = 1/sigma^2 maps spread to NegBin precision
  for (i in 1:N) {
    real phi_i = inv(square(sigma[era[i]]));
    home_score[i] ~ neg_binomial_2_log(mu, phi_i);
    away_score[i] ~ neg_binomial_2_log(mu, phi_i);
  }
}

generated quantities {
  // Posterior predictive draws
  array[N] int home_score_rep;
  array[N] int away_score_rep;

  // Pointwise log-likelihood (for LOO-CV via loo package)
  vector[N] log_lik;

  for (i in 1:N) {
    real phi_i = inv(square(sigma[era[i]]));

    home_score_rep[i] = neg_binomial_2_log_rng(mu, phi_i);
    away_score_rep[i] = neg_binomial_2_log_rng(mu, phi_i);

    log_lik[i] = neg_binomial_2_log_lpmf(home_score[i] | mu, phi_i)
               + neg_binomial_2_log_lpmf(away_score[i] | mu, phi_i);
  }
}
