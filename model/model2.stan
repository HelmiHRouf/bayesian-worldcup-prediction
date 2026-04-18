// Model 2: Continuous-Time Varying Competitive Balance
// 
// Time-varying dispersion of team strengths using a random walk on log-scale.
// This captures how competitive balance evolves continuously across years.
//
// Key quantity of interest: sigma[y] = spread of team strengths in year y.
//   - Large sigma  → teams differ a lot  → less competitive
//   - Small sigma  → teams are similar   → more competitive (parity)
//
// Random walk: log_sigma[y] = log_sigma[y-1] + epsilon[y]
//   where epsilon[y] ~ Normal(0, omega^2)

data {
  int<lower=1> N;                         // number of matches
  int<lower=1> T;                         // number of teams
  int<lower=1> Y;                         // number of years

  array[N] int<lower=1, upper=T> home_team;
  array[N] int<lower=1, upper=T> away_team;
  array[N] int<lower=1, upper=Y> year;    // year index for each match

  array[N] int<lower=0> home_score;
  array[N] int<lower=0> away_score;
}

parameters {
  real mu;                                // baseline log scoring rate

  // Time-varying dispersion (random walk on log scale)
  vector<lower=1e-3>[Y] sigma;           // sigma[y] = exp(log_sigma[y])
  real<lower=1e-3> omega;                // volatility of sigma evolution

  // Team-year specific strengths (random effects, non-centered)
  matrix[T, Y] attack_raw;
  matrix[T, Y] defense_raw;
}

transformed parameters {
  // Non-centered parameterization for computational efficiency
  matrix[T, Y] attack;
  matrix[T, Y] defense;

  for (y in 1:Y) {
    for (t in 1:T) {
      attack[t, y] = sigma[y] * attack_raw[t, y];
      defense[t, y] = sigma[y] * defense_raw[t, y];
    }
  }
}

model {
  // --- Priors ---
  mu ~ normal(0, 1);
  omega ~ normal(0, 0.5);                // half-normal via constraint

  // Random walk on log-sigma (implemented via differences)
  // log_sigma[1] ~ Normal(0, 1) [implicit via sigma[1] prior]
  sigma[1] ~ normal(0, 1);               // initial sigma

  // Differences: log_sigma[y] - log_sigma[y-1] ~ Normal(0, omega)
  for (y in 2:Y) {
    log(sigma[y]) ~ normal(log(sigma[y-1]), omega);
  }

  // Team-year strengths (standard normal, scaled by sigma in transformed params)
  for (y in 1:Y) {
    attack_raw[, y] ~ normal(0, 1);
    defense_raw[, y] ~ normal(0, 1);
  }

  // --- Likelihood ---
  for (i in 1:N) {
    int y = year[i];
    real log_lambda_home = mu
                           + attack[home_team[i], y]
                           - defense[away_team[i], y];

    real log_lambda_away = mu
                           + attack[away_team[i], y]
                           - defense[home_team[i], y];

    home_score[i] ~ poisson_log(log_lambda_home);
    away_score[i] ~ poisson_log(log_lambda_away);
  }
}

generated quantities {
  array[N] int home_score_rep;
  array[N] int away_score_rep;
  vector[N] log_lik;

  // Posterior predictive and log-likelihood for model comparison
  for (i in 1:N) {
    int y = year[i];
    real log_lambda_home = mu
                           + attack[home_team[i], y]
                           - defense[away_team[i], y];

    real log_lambda_away = mu
                           + attack[away_team[i], y]
                           - defense[home_team[i], y];

    home_score_rep[i] = poisson_log_rng(log_lambda_home);
    away_score_rep[i] = poisson_log_rng(log_lambda_away);

    log_lik[i] = poisson_log_lpmf(home_score[i] | log_lambda_home)
                 + poisson_log_lpmf(away_score[i] | log_lambda_away);
  }

  // Extract log_sigma for easier plotting
  vector[Y] log_sigma;
  for (y in 1:Y) {
    log_sigma[y] = log(sigma[y]);
  }
}
