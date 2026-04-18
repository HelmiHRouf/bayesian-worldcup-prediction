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

### Model 1: Shrinking Strength Spread (Poisson Goals)

#### Observed Data

Let $i = 1,\dots,N$ index matches, $t = 1,\dots,T$ index teams, and $e = 1,\dots,E$ index eras.

$y_i^{(H)}$: number of goals scored by the home team in match $i$

$y_i^{(A)}$: number of goals scored by the away team in match $i$

$h(i), a(i) \in \{1,\dots,T\}$: home and away teams in match $i$

$e(i) \in \{1,\dots,E\}$: era of match $i$

------------------------------------------------------------------------

#### Parameters

$\mu$: baseline log scoring rate (global)

$\alpha_{t,e}$: attacking strength of team $t$ in era $e$

$\beta_{t,e}$: defensive strength of team $t$ in era $e$

$\sigma_{\alpha,e}$: dispersion of attacking strengths across all teams in era $e$

$\sigma_{\beta,e}$: dispersion of defensive strengths across all teams in era $e$

## Model

### Likelihood

$$
y_i^{(H)} \sim \text{Poisson}(\lambda_i^{(H)})
$$

$$
y_i^{(A)} \sim \text{Poisson}(\lambda_i^{(A)})
$$

$$
\log \lambda_i^{(H)} = \mu + a_{h(i)} - d_{a(i)}
$$

$$
\log \lambda_i^{(A)} = \mu + a_{a(i)} - d_{h(i)}
$$

where team attacking strengths $a_t$ and defensive strengths $d_t$ are drawn from era-specific distributions:

$$
a_t \sim \mathcal{N}(0, \sigma_{\alpha,e(t)}^2)
$$

$$
d_t \sim \mathcal{N}(0, \sigma_{\beta,e(t)}^2)
$$

------------------------------------------------------------------------


### Priors

**Team-era hierarchy** (each team has era-specific strength):

$$
\alpha_{t,e} \sim \mathcal{N}(0, \sigma_{\alpha,e}^2), \quad t = 1,\dots,T, \quad e = 1,\dots,E
$$

$$
\beta_{t,e} \sim \mathcal{N}(0, \sigma_{\beta,e}^2), \quad t = 1,\dots,T, \quad e = 1,\dots,E
$$

**Era-level dispersion parameters:**

$$
\mu \sim \mathcal{N}(0,1)
$$


**Global parameters:**

### Implementation Note

In the Stan implementation, team-level attacking strengths $a_t$ and defensive strengths $d_t$ are treated as **random effects** (latent variables). Rather than sampling individual team parameters (which would require $2 \times T \times 7$ parameters), we use a **non-centered parameterization with random effects** that marginalizes out the team-level uncertainties during sampling. This yields computationally tractable inference while still estimating the key era-specific dispersion parameters $\sigma_{\alpha,e}$ and $\sigma_{\beta,e}$.

------------------------------------------------------------------------


## Posterior

We infer the joint posterior over era-level dispersion parameters:

$$
p\left(
\mu,
\{\alpha_{t,e}, \beta_{t,e}\}_{t=1,e=1}^{T,E},
\{\sigma_{\alpha,e}, \sigma_{\beta,e}\}_{e=1}^{E}
\mid
\{y_i^{(H)}, y_i^{(A)}\}_{i=1}^{N}
\right)
$$

Note: Team-level strengths $\{a_t, d_t\}_{t=1}^{T}$ are treated as random effects and marginalized out during sampling.

------------------------------------------------------------------------

## Quantity of Interest

The key object of interest is the evolution of

$$
$\{\sigma_{\alpha,e}, \sigma_{\beta,e}\}_{e=1}^{7}
$$

which measures how the **dispersion of team strengths changes across eras**, and therefore captures changes in **competitive balance (parity)** over time.

## Model 2: Continuous-Time Varying Competitive Balance

### Model Setup and Notation

Let $i = 1,\dots,N$ index matches, $t = 1,\dots,T$ index teams, and $y = 1,\dots,Y$ index years.

$h(i), a(i)$: home and away teams in match $i$

$y(i)$: year in which match $i$ is played

------------------------------------------------------------------------

### Observed Data

$y_i^{(H)}$: goals scored by the home team in match $i$

$y_i^{(A)}$: goals scored by the away team in match $i$

------------------------------------------------------------------------

### Parameters

*Global parameters*

$\mu$: baseline log scoring rate

*Team-year latent strengths*

$\alpha_{t,y}$: attacking strength of team $t$ in year $y$

$\beta_{t,y}$: defensive strength of team $t$ in year $y$

*Time-varying competitiveness*

$\sigma_y$: dispersion of team strengths in year $y$

$\omega$: volatility of the evolution of $\sigma_y$

------------------------------------------------------------------------

### Likelihood

$$
y_i^{(H)} \sim \text{Poisson}(\lambda_i^{(H)})
$$

$$
y_i^{(A)} \sim \text{Poisson}(\lambda_i^{(A)})
$$

$$
\log \lambda_i^{(H)} = \mu + \alpha_{h(i),y(i)} - \beta_{a(i),y(i)}
$$

$$
\log \lambda_i^{(A)} = \mu + \alpha_{a(i),y(i)} - \beta_{h(i),y(i)}
$$

------------------------------------------------------------------------

### Team-Level Structure

$$
\alpha_{t,y} \sim \mathcal{N}(0, \sigma_y^2)
$$

$$
\beta_{t,y} \sim \mathcal{N}(0, \sigma_y^2)
$$

------------------------------------------------------------------------

### Evolution of Competitive Balance

We model the evolution of log-dispersion as a random walk:

$$
\log \sigma_y = \log \sigma_{y-1} + \varepsilon_y
$$

$$
\varepsilon_y \sim \mathcal{N}(0, \omega^2)
$$

------------------------------------------------------------------------

### Priors

$$
\sigma_1 \sim \text{HalfNormal}(0,1)
$$

$$
\omega \sim \text{HalfNormal}(0,0.5)
$$

$$
\mu \sim \mathcal{N}(0,1)
$$

------------------------------------------------------------------------

### Posterior

We infer the joint posterior:

$$
p\left(
\mu,
\{\alpha_{t,y}, \beta_{t,y}\},
\{\sigma_y\},
\omega
\mid
\{y_i^{(H)}, y_i^{(A)}\}
\right)
$$

------------------------------------------------------------------------

### Quantity of Interest

The primary object of interest is the trajectory

$$
\{\sigma_y\}_{y=1}^{Y}
$$

which represents the *evolution of competitive balance over time*.

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
