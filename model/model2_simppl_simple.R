# =============================================================================
# model2_simppl_simple.R — Simplified Model 2 for simPPLe
# 
# This is a SIMPLIFIED version of Model 2 that works with simPPLe's 
# importance sampling. The full Model 2 (~11,400+ latent variables) is
# too high-dimensional for naive importance sampling.
#
# Simplifications:
# - 7 eras instead of ~27 years (reduces parameters by ~4x)
# - Team strengths marginalized (only era-level sigma tracked)
# - Returns scalar quantity (final sigma) instead of full trajectory
# =============================================================================

# Load simPPLe scaffold
source("model/simppl_scaffold.R")

# =============================================================================
# 1. Load and Aggregate Data by Era (like Model 1)
# =============================================================================

load_model2_era_data <- function() {
  #' Load data and aggregate into 7 eras (matching Model 1 structure)
  
  matches <- read.csv("data/processed/matches.csv")
  
  # Define eras (same as Model 1)
  matches$era <- cut(matches$year, 
                       breaks = c(1997, 2002, 2006, 2010, 2014, 2018, 2022, 2026),
                       labels = 1:7)
  
  # Aggregate matches by era
  era_stats <- matches %>%
    group_by(era) %>%
    summarise(
      total_matches = n(),
      avg_home_score = mean(home_score),
      avg_away_score = mean(away_score),
      .groups = 'drop'
    )
  
  # Return era-level aggregated data
  data <- list(
    N = nrow(matches),
    E = 7,  # 7 eras
    era = as.integer(matches$era),
    home_score = matches$home_score,
    away_score = matches$away_score,
    era_stats = era_stats,
    years = c(2000, 2004, 2008, 2012, 2016, 2020, 2024)  # era midpoints
  )
  
  return(data)
}

# =============================================================================
# 2. Simplified Model 2: Random Walk on Log-Sigma (Era Level)
# =============================================================================

#' Simplified Model 2 in simPPLe
#' 
#' Instead of team × year parameters, we model:
#' - Random walk on log_sigma across eras (7 values instead of 5700+)
#' - Poisson likelihood with era-specific global rates

make_model2_era_function <- function(data, return_final_sigma = TRUE) {
  
  N <- data$N
  E <- data$E
  
  function() {
    # --- Global parameters ---
    mu <- simulate(Norm(mean = 0, sd = 1))
    
    # Random walk on log_sigma across eras
    log_sigma_1 <- simulate(Norm(mean = 0, sd = 1))
    omega <- max(0.001, simulate(Exp(rate = 2)))  # volatility
    
    # Build random walk for 7 eras
    log_sigma <- numeric(E)
    log_sigma[1] <- log_sigma_1
    
    for (e in 2:E) {
      increment <- simulate(Norm(mean = 0, sd = 1))
      log_sigma[e] <- log_sigma[e-1] + omega * increment
    }
    
    # Convert to sigma
    sigma <- exp(log_sigma)
    
    # --- Likelihood ---
    # Simplified: era-specific Poisson with sigma as dispersion factor
    for (i in 1:N) {
      e <- data$era[i]
      
      # Era-specific rates (simplified model)
      lambda_home <- exp(mu + 0.1 * sigma[e])  # small sigma effect
      lambda_away <- exp(mu - 0.1 * sigma[e])
      
      observe(data$home_score[i], Pois(lambda = max(0.01, lambda_home)))
      observe(data$away_score[i], Pois(lambda = max(0.01, lambda_away)))
    }
    
    # Return scalar for importance sampling stability
    if (return_final_sigma) {
      return(sigma[E])  # Return 2024 era sigma
    } else {
      return(mean(sigma))  # Return average sigma
    }
  }
}

# =============================================================================
# 3. Run simPPLe Inference
# =============================================================================

run_model2_simppl <- function(num_iterations = 1000) {
  
  message("Loading era-aggregated data...")
  data <- load_model2_era_data()
  message(sprintf("Data: %d matches across %d eras", data$N, data$E))
  
  # Create model function
  model_fn <- make_model2_era_function(data, return_final_sigma = TRUE)
  
  message(sprintf("Running simPPLe importance sampling with %d iterations...", num_iterations))
  message("Model: Simplified era-level random walk on sigma")
  
  # Run inference
  result <- posterior(model_fn, num_iterations)
  
  # Compute diagnostics
  ess <- compute_ess(result)
  samples <- extract_samples(result)
  
  message(sprintf("Effective Sample Size: %.1f / %d (%.1f%%)", 
                  ess, num_iterations, 100 * ess / num_iterations))
  message(sprintf("Posterior mean of sigma (2024): %.4f", samples$weighted_mean))
  message(sprintf("Posterior sd of sigma (2024): %.4f", samples$weighted_sd))
  
  # Return full analysis
  return(list(
    posterior_mean = result$expectation,
    samples = samples,
    ess = ess,
    ess_ratio = ess / num_iterations,
    data = data,
    raw_result = result
  ))
}

# =============================================================================
# 4. Visualization
# =============================================================================

plot_sigma_posterior <- function(result) {
  #' Plot posterior of final era sigma
  
  samples <- result$samples
  
  if (require("ggplot2", quietly = TRUE)) {
    df <- data.frame(
      sigma_2024 = samples$values,
      weight = samples$weights
    )
    
    p <- ggplot(df, aes(x = sigma_2024)) +
      geom_histogram(aes(weight = weight), bins = 30, 
                     fill = "steelblue", color = "white", alpha = 0.8) +
      geom_vline(xintercept = samples$weighted_mean, 
                 color = "red", linetype = "dashed", linewidth = 1) +
      labs(
        title = "Posterior Distribution of Sigma (2024 Era)",
        subtitle = sprintf("simPPLe Importance Sampling (ESS = %.1f)", result$ess),
        x = expression(sigma[2024]),
        y = "Density"
      ) +
      theme_minimal(base_size = 12) +
      theme(
        plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(color = "gray40")
      )
    
    print(p)
    ggsave("figures/model2_simppl_sigma_2024.png", p, width = 8, height = 6, dpi = 150)
    message("Saved plot to figures/model2_simppl_sigma_2024.png")
  } else {
    # Base R plot
    weighted.hist <- function(x, w, ...) {
      breaks <- pretty(range(x), n = 30)
      h <- hist(x, breaks = breaks, plot = FALSE)
      counts <- sapply(1:(length(breaks)-1), function(i) {
        sum(w[x >= breaks[i] & x < breaks[i+1]])
      })
      h$density <- counts / sum(counts) / diff(breaks)
      plot(h, freq = FALSE, ...)
    }
    
    weighted.hist(samples$values, samples$weights,
                  main = "Posterior of Sigma (2024)",
                  xlab = expression(sigma[2024]),
                  col = "steelblue", border = "white")
    abline(v = samples$weighted_mean, col = "red", lty = 2, lwd = 2)
  }
}

# =============================================================================
# Example Usage
# =============================================================================

if (interactive()) {
  message("=" ,rep("=", 60))
  message("Model 2 (Simplified) - simPPLe Implementation")
  message("=" ,rep("=", 60))
  message("")
  message("This is a SIMPLIFIED version for simPPLe with:")
  message("  - 7 eras instead of 27 years")
  message("  - ~7 parameters instead of 11,400+")
  message("  - Returns scalar (final sigma) instead of full trajectory")
  message("")
  message("To run:")
  message("  source('model/model2_simppl_simple.R')")
  message("  result <- run_model2_simppl(num_iterations = 1000)")
  message("  plot_sigma_posterior(result)")
  message("")
  message("For full Model 2, use Stan (HMC) instead!")
}
