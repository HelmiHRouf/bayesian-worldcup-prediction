# Bayesian World Cup 2026 Prediction

Predicting the probability of each national team winning the 2026 FIFA World Cup using Bayesian inference.

## Team Members

- **Blasius Halley Theo Boniarga**
- **Helmi Hidayat Rouf**
- **Louis Charistias Saragih**

## Project Overview

Football match outcomes are inherently uncertain and depend on latent team strength, making them well suited for probabilistic modeling. Instead of predicting a single winner, our Bayesian approach estimates posterior distributions of team strengths and tournament outcomes.

### Research Question

> What is the probability that each national team will win the 2026 FIFA World Cup given historical match data?

We will:
1. Infer latent team strength parameters from historical match results.
2. Model match outcomes probabilistically.
3. Simulate the tournament to estimate posterior winning probabilities.

## Datasets

### Main Dataset: International Football Results

- **Source:** [International Football Results (1872–present)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- **Description:** International football match results including date, teams, scores, and tournament type.

| date       | home_team | away_team    | home_score | away_score | tournament     |
|------------|-----------|--------------|------------|------------|----------------|
| 2022-11-21 | England   | Iran         | 6          | 2          | FIFA World Cup |
| 2022-11-22 | Argentina | Saudi Arabia | 1          | 2          | FIFA World Cup |

### Backup Dataset: FIFA Rankings

- **Source:** [FIFA International Soccer Men's Ranking](https://www.kaggle.com/datasets/tadhgfitzgerald/fifa-international-soccer-mens-ranking-1993now)
- **Description:** Team ratings over time, usable as additional covariates or priors for team strength.

| date       | team   | rank | rating |
|------------|--------|------|--------|
| 2024-01-01 | France | 2    | 1840   |
| 2024-01-01 | Brazil | 3    | 1834   |

## Methodology

### Model 1: Baseline Bayesian Bradley-Terry Model

Each team has a latent strength parameter θ. Match outcome probability:

```
P(Team A wins) = σ(θ_A − θ_B)
```

- θ_i ∼ N(0, 1)
- σ = logistic function

Assumes constant team strength; matches depend only on relative strength.

### Model 2: Hierarchical Bayesian Goal Model

Goals modeled using Poisson distributions:

```
Goals_home ∼ Poisson(λ_home)
log(λ_home) = attack_home − defense_away + home_advantage
```

Parameters per team: attack strength, defensive strength, plus a shared home advantage parameter. Hierarchical priors are used for team parameters.

### Posterior Computation

| Method | Technique | Tool |
|--------|-----------|------|
| Method 1 | Markov Chain Monte Carlo (MCMC) | Stan |
| Method 2 | Variational Inference (Stochastic Gradient) | — |

### Model Comparison

- Predictive performance on held-out matches
- Posterior predictive checks
- KL divergence between posterior approximations

### Tournament Simulation

Monte Carlo simulations (e.g., 10,000 runs) to estimate:
- Probability each team wins the tournament
- Probability of reaching each round

## Team Contribution Plan

| Member | Responsibilities |
|--------|-----------------|
| Student 1 | Data collection & preprocessing, baseline model implementation |
| Student 2 | Hierarchical Bayesian model, Stan model development |
| Student 3 | Posterior inference methods, model comparison & tournament simulation |

All members contribute to report writing, visualization, and interpretation.

## Project Structure

```
bayesian-worldcup-prediction/
├── data/
│   ├── raw/                # Original unprocessed datasets
│   └── processed/          # Cleaned and transformed data
├── R/                      # R helper functions and utilities
├── stan/                   # Stan model files (.stan)
├── scripts/                # R scripts for data processing, fitting, simulation
├── rmd/                    # R Markdown notebooks for exploration and reporting
├── figures/                # Generated figures for the report
├── report/                 # Final report source files
├── .gitignore
├── bayesian-worldcup-prediction.Rproj
└── README.md
```

## Getting Started

### Prerequisites

- [R](https://cran.r-project.org/) (≥ 4.2)
- [RStudio](https://posit.co/download/rstudio-desktop/) (recommended)
- [CmdStan](https://mc-stan.org/cmdstanr/articles/cmdstanr.html) (installed via `cmdstanr`)

### Installation

```bash
git clone https://github.com/HelmiHRouf/bayesian-worldcup-prediction.git
cd bayesian-worldcup-prediction
```

Then open the `.Rproj` file in RStudio and install the required packages:

```r
install.packages(c(
  "tidyverse", "rstan", "cmdstanr", "bayesplot",
  "loo", "posterior", "ggplot2", "knitr", "rmarkdown"
))

# Install CmdStan (one-time setup)
cmdstanr::install_cmdstan()
```

## License

This project is developed for academic purposes.
