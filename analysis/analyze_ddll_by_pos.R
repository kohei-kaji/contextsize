library(readr)
library(tidyverse)
library(dplyr)
library(brms)


ddll_by_pos <- function(df, bayesian = FALSE){
  # group the ddll by pos tags
  df_filtered <- df %>%
    group_by(pos) %>%
    mutate(pos_freq = n()) %>%
    filter(pos_freq > 50) %>%
    ungroup() %>%
    mutate(scaled_ddll = as.numeric(scale(ddll)))
  
  # fit the regression model (frequentist or bayesian)
  if (!bayesian) {
    m_pos <- lm(scaled_ddll ~ pos - 1, data = df_filtered)
    m_coef <- summary(m_pos)$coefficients
    m_ci   <- confint(m_pos)
    
    model_results <- data.frame(
      term      = rownames(m_coef),
      estimate  = m_coef[, "Estimate"],
      conf.low  = m_ci[, 1],
      conf.high = m_ci[, 2],
      stringsAsFactors = FALSE
    ) 
  } else {
    m_pos_brm <- brm(
      formula = scaled_ddll ~ pos - 1,
      data = df_filtered,
      family = gaussian(),
      chains = 4,
      cores = 4,
      iter = 2000,
      seed = 1
    )
    m_coef <- fixef(m_pos_brm)
    model_results <- data.frame(
      term      = rownames(m_coef),
      estimate  = m_coef[, "Estimate"],
      conf.low  = m_coef[, "Q2.5"],
      conf.high = m_coef[, "Q97.5"],
      stringsAsFactors = FALSE
    )
  }
  
  # final cleaning of the dataframe before plotting
  pos_results <- model_results %>%
    mutate(pos = str_remove(term, "^pos")) %>%
    arrange(desc(estimate)) %>%
    mutate(pos = factor(pos, levels = pos))
  
  return(pos_results)
}

plot_pos_effect <- function (pos_results, title_text) {
  p_pos <- ggplot(pos_results, aes(x = estimate, y = pos)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "#63666A") +
    geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2, color = "#041E42") +
    geom_point(shape = 1, size = 3, color = "#041E42", stroke = 1) +  
    labs(
      title = title_text,
      x = "Delta Delta Log-Likelihood (Scaled)",
      y = "Part of Speech (POS)"
    ) +
    theme_bw() +
    theme(
      axis.text.y = element_text(size = 10),
      plot.title  = element_text(face = "bold")
    )
  
  p_pos
}

