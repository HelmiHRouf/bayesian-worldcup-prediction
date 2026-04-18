// Model 1: Shrinking Strength Spread (Poisson score)
//
// Team-level attack/defense effects are marginalized out via Negative Binomial.
// NegBin(mu, phi) ≈ Poisson-LogNormal where phi = 1/sigma^2.
// Key quantity of interest: sigma[e] = spread of team strengths per era.
//   - Large sigma  → teams differ a lot  → less competitive
//   - Small sigma  → teams are similar   → more competitive (parity)
// haloo
data {
  int<lower=1> N;                         // matches
  int<lower=1> T;                         // teams
  int<lower=1> E;                         // eras

  array[N] int<lower=1, upper=T> home_team;
  array[N] int<lower=1, upper=T> away_team;

  array[T] int<lower=1, upper=E> team_era;   // era of each team

  array[N] int<lower=0> home_score;
  array[N] int<lower=0> away_score;
}

parameters {
  real mu;

  vector[T] attack;
  vector[T] defense;

  vector<lower=1e-3>[E] sigma_att;
  vector<lower=1e-3>[E] sigma_def;
}

model {
  // --- Priors ---
  mu ~ normal(0, 1);

  sigma_att ~ normal(0, 1);
  sigma_def ~ normal(0, 1);

  // --- Hierarchical team strengths ---
  for (t in 1:T) {
    attack[t] ~ normal(0, sigma_att[team_era[t]]);
    defense[t] ~ normal(0, sigma_def[team_era[t]]);
  }

  // --- Likelihood ---
  for (i in 1:N) {
    real log_lambda_home = mu
                           + attack[home_team[i]]
                           - defense[away_team[i]];

    real log_lambda_away = mu
                           + attack[away_team[i]]
                           - defense[home_team[i]];

    home_score[i] ~ poisson_log(log_lambda_home);
    away_score[i] ~ poisson_log(log_lambda_away);
  }
}

generated quantities {
  array[N] int home_score_rep;
  array[N] int away_score_rep;
  vector[N] log_lik;

  for (i in 1:N) {
    real log_lambda_home = mu
                           + attack[home_team[i]]
                           - defense[away_team[i]];

    real log_lambda_away = mu
                           + attack[away_team[i]]
                           - defense[home_team[i]];

    home_score_rep[i] = poisson_log_rng(log_lambda_home);
    away_score_rep[i] = poisson_log_rng(log_lambda_away);

    log_lik[i] =
      poisson_log_lpmf(home_score[i] | log_lambda_home) +
      poisson_log_lpmf(away_score[i] | log_lambda_away);
  }
}
