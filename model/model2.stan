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
  // Reparameterized: log_sigma is the parameter, sigma is derived
  vector[Y] log_sigma_raw;               // non-centered random walk increments
  real<lower=-3, upper=3> log_sigma_1;   // initial log_sigma[1] (~ 0.05 to 20)
  real<lower=1e-3> omega;      // volatility of sigma evolution (constrained)

  // Team-year specific strengths (random effects, non-centered)
  matrix[T, Y] attack_raw;
  matrix[T, Y] defense_raw;

  real<lower=0, upper=1> rho; // AR(1)
}

transformed parameters {
  vector[Y] log_sigma;
  vector<lower=0>[Y] sigma;
  matrix[T, Y] attack;
  matrix[T, Y] defense;

  // Reconstruct random walk on log_sigma
  log_sigma[1] = log_sigma_1;
  for (y in 2:Y) {
    log_sigma[y] = log_sigma[y-1] + omega * log_sigma_raw[y];
  }

  // Exponentiate to get sigma
  sigma = exp(log_sigma);

  // Year 1: non-centered
  for (t in 1:T) {
    attack[t, 1] = sigma[1] * attack_raw[t, 1];
    defense[t, 1] = sigma[1] * defense_raw[t, 1];
  }
  // Subsequent years: AR(1) evolution
  for (y in 2:Y) {
    for (t in 1:T) {
      attack[t, y] = rho * attack[t, y-1] + sigma[y] * attack_raw[t, y];
      defense[t, y] = rho * defense[t, y-1] + sigma[y] * defense_raw[t, y];
    }
  }
}

model {
  // --- Priors ---
  rho ~ beta(8, 2);

  mu ~ normal(0, 1);
  // soft prior for log_sigma_1 (centered on ~1.0, sigma ~2.7)
  log_sigma_1 ~ normal(0.5, 1);
  // tighter prior on omega to prevent explosive random walk
  omega ~ exponential(2);                // mean ~0.5, sd ~0.5

  // Random walk increments: standard normal (scaled by omega in transformed params)
  log_sigma_raw ~ normal(0, 1);

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

  // log_sigma is already computed in transformed parameters
}
