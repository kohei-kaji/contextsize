library(readr)
library(dplyr)
library(tidyverse)
library(data.table)
library(stats)

ns_surp_base <- "data/ns_surp"
baseline_df <- read_csv("data/baselines_ns.csv", show_col_types = FALSE)
rt_path <- 'data/naturalstories/naturalstories_RTS/processed_RTs.tsv'

# create a new dataframe with delta log-likelihoods for both context lengths and their delta delta log-likelihood
compare_two_contexts <- function(
    df,
    rt_path,
    target_model,
    context_a,
    context_b,
    n_folds = 10
){
  # assign the correct context names
  ctxs <- sort(c(context_a, context_b))
  short_ctx <- ctxs[1]
  long_ctx <- ctxs[2]
  
  model_clean <- gsub('[-.]', '_', target_model)  
  
  # add the surprisal values to the baseline file
  for (ctx in ctxs){
    fpath <- sprintf('%s/%s/context_%d.txt', ns_surp_base, model_clean, ctx)
    if (file.exists(fpath)){
      vals <- scan(fpath, what = numeric(), quiet = TRUE)
      if (length(vals) != nrow(df)){
        warning(sprintf('row mismatch for %s ctx%d: expected %d, got %d — skipping', model_clean, ctx, nrow(df), length(vals)))
        next
      }
      df[[sprintf('%s_ctx%d', model_clean, ctx)]] <- vals
    }else{
      warning(sprintf('File not found: %s', fpath))
    }
  }

  
  # taking care of spillovers
  no_spill_cols <- c("story", "zone", "word", "bos", "eos", "is_punct", "position")
  spill_cols <- setdiff(names(df), no_spill_cols)
  
  df_so1 <- df %>%
    mutate(zone = zone + 1) %>%
    select(story, zone, all_of(spill_cols)) %>%
    rename_with(~ paste0(., '_so1'), .cols = all_of(spill_cols))

  print(names(df_so1))
  df_so2 <- df %>%
    mutate(zone = zone + 2) %>%
    select(story, zone, all_of(spill_cols)) %>%
    rename_with(~ paste0(., '_so2'), .cols = all_of(spill_cols))
  print(names(df_so2))
  
  df <- df %>%
    left_join(df_so1, by = c('story', 'zone')) %>%
    left_join(df_so2, by = c('story', 'zone')) %>%
    filter(is_punct == 0, bos == 0, eos == 0)%>%
    select(-is_punct, -bos, -eos) %>%
    drop_na()
  
  # add RT to the df
  rts_summary <- read.table(rt_path, sep = "\t", quote = "", header = TRUE) %>%
    rename(story = item) %>%
    group_by(story, zone) %>%
    summarise(mean_RT = mean(RT, na.rm = TRUE), .groups = 'drop')
  
  df <- merge(rts_summary, df, by = c("story", "zone"), sort = FALSE)

  
  # build the required columns
  short_col     <- sprintf("%s_ctx%d",     model_clean, short_ctx)
  short_col_so1 <- sprintf("%s_ctx%d_so1", model_clean, short_ctx)
  short_col_so2 <- sprintf("%s_ctx%d_so2", model_clean, short_ctx)
  
  long_col      <- sprintf("%s_ctx%d",     model_clean, long_ctx)
  long_col_so1  <- sprintf("%s_ctx%d_so1", model_clean, long_ctx)
  long_col_so2  <- sprintf("%s_ctx%d_so2", model_clean, long_ctx)

  required_cols <- c(
    "mean_RT", "story", "zone", "position", "word", "pos", "wlen", "wlen_so1", "wlen_so2",
    "unisurp", "unisurp_so1", "unisurp_so2",
    short_col, short_col_so1, short_col_so2,
    long_col,  long_col_so1,  long_col_so2
  )
  
  # make sure that the required columns exist
  missing_cols <- setdiff(required_cols, names(df))
  
  if(length(missing_cols) > 0){
    stop('missing columns:', paste(missing_cols, collapse = ', '))
  }
  
  # scaling for numeric stability
  scale_cols <- grep("^(zone|position|wlen|unisurp)|_ctx\\d+", names(df), value = TRUE, perl = TRUE)
  df[scale_cols] <- lapply(df[scale_cols], function(x) as.numeric(scale(x)))
  
  
  # create CV folds
  set.seed(1)
  folds <- sample(rep(seq_len(n_folds), length.out = nrow(df)))
  idx_list <- lapply(seq_len(n_folds), function(k) {
    list(
      tr = which(folds != k),
      te = which(folds == k)
    )
  })
  
  # define formulas 
  base_formula_str <- paste(
    "mean_RT ~ zone + position +",
    paste(c(
      "wlen", "wlen_so1", "wlen_so2",
      "unisurp", "unisurp_so1", "unisurp_so2"
    ), collapse = " + ")
  )
  
  short_formula_str <- paste(
    "mean_RT ~ zone + position +",
    paste(c(
      "wlen", "wlen_so1", "wlen_so2",
      "unisurp", "unisurp_so1", "unisurp_so2",
      short_col, short_col_so1, short_col_so2
    ), collapse = " + ")
  )
  
  long_formula_str <- paste(
    "mean_RT ~ zone + position +",
    paste(c(
      "wlen", "wlen_so1", "wlen_so2",
      "unisurp", "unisurp_so1", "unisurp_so2",
      long_col, long_col_so1, long_col_so2
    ), collapse = " + ")
  )
  
  base_formula  <- as.formula(base_formula_str)
  short_formula <- as.formula(short_formula_str)
  long_formula  <- as.formula(long_formula_str)
  
  # allocate storage for per-word held-out log-likelihoods
  ll_base  <- rep(NA_real_, nrow(df))
  ll_short <- rep(NA_real_, nrow(df))
  ll_long  <- rep(NA_real_, nrow(df))

  # fold loop
  for (k in seq_len(n_folds)){
    tr <- df[idx_list[[k]]$tr, ]
    te <- df[idx_list[[k]]$te, ]
    y <- te$mean_RT
    te_idx <- idx_list[[k]]$te
    
    # baseline model
    m_base <- lm(base_formula, data = tr)
    ll_base[te_idx] <- dnorm(
      y,
      mean = predict(m_base, newdata = te),
      sd   = sigma(m_base),
      log  = TRUE
    )
    
    # short-context model
    m_short <- lm(short_formula, data = tr)
    ll_short[te_idx] <- dnorm(
      y,
      mean = predict(m_short, newdata = te),
      sd   = sigma(m_short),
      log  = TRUE
    )
    
    # long-context model
    m_long <- lm(long_formula, data = tr)
    ll_long[te_idx] <- dnorm(
      y,
      mean = predict(m_long, newdata = te),
      sd   = sigma(m_long),
      log  = TRUE
    )
  }
  
    # compute delta and delta-delta log-likelihood
    dll_short <- ll_short - ll_base
    dll_long  <- ll_long  - ll_base
    ddll      <- ll_long  - ll_short
  
  
    # return the df
    df <- df %>%
      mutate(dll_short = dll_short,
             dll_long = dll_long, 
             ddll = ddll
             )
    
    return(df)
}