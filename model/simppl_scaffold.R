# =============================================================================
# simppl_scaffold.R — simPPLe Probabilistic Programming Framework
# Based on STAT 405 course materials
# =============================================================================

suppressPackageStartupMessages(library(distr))

## Utilities to make the distr library a bit nicer to use

p <- function(distribution, realization) {
  d(distribution)(realization) # return the PMF or density 
}

Bern = function(probability_to_get_one) {
  DiscreteDistribution(supp = 0:1, prob = c(1-probability_to_get_one, probability_to_get_one))
}

## Key functions called by simPPLe programs

# Use simulate(distribution) for unobserved random variables
simulate <- function(distribution) {
  r(distribution)(1) # sample once from the given distribution
}

# Use observe(realization, distribution) for observed random variables
observe = function(realization, distribution) {
  # `<<-` lets us modify variables that live in the global scope from inside a function
  weight <<- weight * p(distribution, realization) 
}

## Posterior computation using importance sampling

posterior = function(ppl_function, number_of_iterations) {
  numerator = 0.0             
  denominator = 0.0
  samples = list()
  
  for (i in 1:number_of_iterations) {
    weight <<- 1.0                    
    g_i = ppl_function()
    numerator = numerator + weight * g_i
    denominator = denominator + weight
    samples[[i]] = list(weight = weight, value = g_i)
  }
  
  result = list(
    expectation = numerator/denominator,
    samples = samples,
    denominator = denominator
  )
  return(result)
}

## Effective Sample Size computation

compute_ess = function(posterior_result) {
  samples = posterior_result$samples
  weights = sapply(samples, function(s) s$weight)
  
  # ESS = (sum w_i)^2 / sum(w_i^2)
  ess = sum(weights)^2 / sum(weights^2)
  return(ess)
}

## Extract weighted samples for a specific quantity

extract_samples = function(posterior_result, quantity_name = "value") {
  samples = posterior_result$samples
  weights = sapply(samples, function(s) s$weight)
  values = sapply(samples, function(s) s$value)
  
  # Normalize weights
  normalized_weights = weights / sum(weights)
  
  return(list(
    values = values,
    weights = normalized_weights,
    weighted_mean = sum(values * normalized_weights),
    weighted_sd = sqrt(sum(normalized_weights * (values - sum(values * normalized_weights))^2))
  ))
}

message("simPPLe scaffold loaded successfully!")
message("Available functions: simulate(), observe(), posterior(), compute_ess(), extract_samples()")
