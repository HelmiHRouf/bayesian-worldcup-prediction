# Bayesian World Cup 2026 Prediction

Predicting the probability of each national team winning the 2026 FIFA World Cup using Bayesian inference.

## Team Members

-   **Blasius Halley Theo Boniarga**
-   **Helmi Hidayat Rouf**
-   **Louis Charistias Saragih**

## Project Overview

Football match outcomes are inherently uncertain and depend on latent team strength, making them well suited for probabilistic modeling. Instead of predicting a single winner, our Bayesian approach estimates posterior distributions of team strengths and tournament outcomes.

### Research Question

> Has international football become more unpredictable over time?

We will: 1. Model how team strength distributions have evolved across historical eras. 2. Quantify changes in match outcome randomness using Bayesian hierarchical models. 3. Compare posterior estimates of competitiveness across decades.

## Datasets

### Main Dataset: International Football Results

-   **Source:** [International Football Results (1872--present)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
-   **Description:** International football match results including date, teams, scores, and tournament type.

| date       | home_team | away_team    | home_score | away_score | tournament     |
|------------|-----------|--------------|------------|------------|----------------|
| 2022-11-21 | England   | Iran         | 6          | 2          | FIFA World Cup |
| 2022-11-22 | Argentina | Saudi Arabia | 1          | 2          | FIFA World Cup |

### Backup Dataset: FIFA Rankings

-   **Source:** [FIFA International Soccer Men's Ranking](https://www.kaggle.com/datasets/tadhgfitzgerald/fifa-international-soccer-mens-ranking-1993now)
-   **Description:** Team ratings over time, usable as additional covariates or priors for team strength.

| date       | team   | rank | rating |
|------------|--------|------|--------|
| 2024-01-01 | France | 2    | 1840   |
| 2024-01-01 | Brazil | 3    | 1834   |

## Methodology

### Model 1: Baseline Bayesian Bradley-Terry Model

#### Observed Data

Let $i = 1,\dots,N$ index matches, $t = 1,\dots,T$ index teams, and $e = 1,\dots,E$ index eras.

$y_i^{(H)}$: number of goals scored by the home team in match $i$

$y_i^{(A)}$: number of goals scored by the away team in match $i$

$h(i), a(i) \in \{1,\dots,T\}$: home and away teams in match $i$

$e(i) \in \{1,\dots,E\}$: era of match $i$

------------------------------------------------------------------------

#### Parameters

$\mu$: baseline log scoring rate

$\delta$: home advantage

$\alpha_t$: attacking strength of team $t$

$\beta_t$: defensive strength of team $t$

$\sigma_{\alpha,e}$: dispersion of attacking strengths in era $e$

$\sigma_{\beta,e}$: dispersion of defensive strengths in era $e$

## Model

### Likelihood

$$
y_i^{(H)} \sim \text{Poisson}(\lambda_i^{(H)})
$$

$$
y_i^{(A)} \sim \text{Poisson}(\lambda_i^{(A)})
$$

$$
\log \lambda_i^{(H)} = \mu + \delta + \alpha_{h(i)} - \beta_{a(i)}
$$

$$
\log \lambda_i^{(A)} = \mu + \alpha_{a(i)} - \beta_{h(i)}
$$

------------------------------------------------------------------------

### Priors

**Team-level hierarchy**

$$
\alpha_t \sim \mathcal{N}(0, \sigma_{\alpha,e(t)}^2)
$$

$$
\beta_t \sim \mathcal{N}(0, \sigma_{\beta,e(t)}^2)
$$

**Era-level dispersion parameters**

$$
\sigma_{\alpha,e} \sim \text{HalfNormal}(0,1), \quad e = 1,\dots,E
$$

$$
\sigma_{\beta,e} \sim \text{HalfNormal}(0,1), \quad e = 1,\dots,E
$$

**Global parameters**

$$
\mu \sim \mathcal{N}(0,1)
$$

$$
\delta \sim \mathcal{N}(0,0.5^2)
$$

------------------------------------------------------------------------

### Identifiability Constraints (optional)

$$
\sum_{t=1}^{T} \alpha_t = 0
$$

$$
\sum_{t=1}^{T} \beta_t = 0
$$

------------------------------------------------------------------------

## Posterior

We infer the joint posterior:

$$
p\left(
\mu, \delta,
\{\alpha_t, \beta_t\}_{t=1}^{T},
\{\sigma_{\alpha,e}, \sigma_{\beta,e}\}_{e=1}^{E}
\mid
\{y_i^{(H)}, y_i^{(A)}\}_{i=1}^{N}
\right)
$$

------------------------------------------------------------------------

## Quantity of Interest

The key object of interest is the evolution of

$$
\{\sigma_{\alpha,e}, \sigma_{\beta,e}\}_{e=1}^{E}
$$

which measures how the **dispersion of team strengths changes across eras**, and therefore captures changes in **competitive balance (parity)** over time.

### Model 2: Hierarchical Bayesian Goal Model

Goals modeled using Poisson distributions:

```         
Goals_home ∼ Poisson(λ_home)
log(λ_home) = attack_home − defense_away + home_advantage
```

Parameters per team: attack strength, defensive strength, plus a shared home advantage parameter. Hierarchical priors are used for team parameters.

### Posterior Computation

| Method   | Technique                                   | Tool |
|----------|---------------------------------------------|------|
| Method 1 | Markov Chain Monte Carlo (MCMC)             | Stan |
| Method 2 | Variational Inference (Stochastic Gradient) | ---  |

### Model Comparison

-   Predictive performance on held-out matches
-   Posterior predictive checks
-   KL divergence between posterior approximations

### Tournament Simulation

Monte Carlo simulations (e.g., 10,000 runs) to estimate: - Probability each team wins the tournament - Probability of reaching each round

## Team Contribution Plan

| Member | Responsibilities |
|--------------------------|----------------------------------------------|
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

-   [R](https://cran.r-project.org/) (≥ 4.2)
-   [RStudio](https://posit.co/download/rstudio-desktop/) (recommended)
-   [CmdStan](https://mc-stan.org/cmdstanr/articles/cmdstanr.html) (installed via `cmdstanr`)

### Installation

``` bash
git clone https://github.com/HelmiHRouf/bayesian-worldcup-prediction.git
cd bayesian-worldcup-prediction
```

Then open the `.Rproj` file in RStudio and install the required packages:

``` r
install.packages(c(
  "tidyverse", "rstan", "cmdstanr", "bayesplot",
  "loo", "posterior", "ggplot2", "knitr", "rmarkdown"
))

# Install CmdStan (one-time setup)
cmdstanr::install_cmdstan()
```

## License

This project is developed for academic purposes.
