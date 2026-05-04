library(readr)
library(dplyr)
library(tidyverse)
library(data.table)
library(stats)

set.seed(1)

ns_surp_base <- "../data/ns_surp"
baseline <- read_csv("../data/baselines_ns.csv", show_col_types = FALSE)

target_model <- "gpt2"
target_contexts <- c(2, 100)

loaded <- 0
for (ctx in target_contexts) {
  fpath <- sprintf("%s/%s/context_%d.txt", ns_surp_base, target_model, ctx)
  if (file.exists(fpath)) {
    vals  <- scan(fpath, what = numeric(), quiet = TRUE)
    if (length(vals) != nrow(baseline)) {
      warning(sprintf("Row mismatch for %s ctx%d: expected %d, got %d — skipping", target_model, ctx, nrow(baseline), length(vals)))
      next
    }
    baseline[[sprintf("%s_ctx%d", target_model, ctx)]] <- vals
    loaded <- loaded + 1
  } else {
    warning(sprintf("File not found: %s", fpath))
  }
}

pred <- baseline
no_spill_cols <- c("story", "zone", "word", "bos", "eos", "is_punct", "position")
spill_cols    <- setdiff(names(pred), no_spill_cols)

pred_so1 <- pred %>%
  mutate(zone = zone + 1) %>%
  select(story, zone, all_of(spill_cols)) %>%
  rename_with(~ paste0(., "_so1"), .cols = all_of(spill_cols))

pred_so2 <- pred %>%
  mutate(zone = zone + 2) %>%
  select(story, zone, all_of(spill_cols)) %>%
  rename_with(~ paste0(., "_so2"), .cols = all_of(spill_cols))

pred <- pred %>%
  merge(pred_so1, by = c("story", "zone"), sort = FALSE) %>%
  merge(pred_so2, by = c("story", "zone"), sort = FALSE) %>%
  filter(is_punct == 0, bos == 0, eos == 0) %>%
  select(-is_punct, -bos, -eos) %>%
  drop_na()


rt_path <- "../data/naturalstories/naturalstories_RTS/processed_RTs.tsv"
rts_summary <- read.table(rt_path, sep = "\t", quote = "", header = TRUE) %>%
  select(WorkerId, item, zone, RT) %>%
  rename(story = item) %>%
  group_by(story, zone) %>%
  summarise(mean_RT = mean(RT, na.rm = TRUE), .groups = "drop")

df <- merge(rts_summary, pred, by = c("story", "zone"), sort = FALSE)

scale_cols <- grep("^(zone|position|wlen|unisurp)|_ctx\\d+", names(df), value = TRUE, perl = TRUE)
df[scale_cols] <- lapply(df[scale_cols], function(x) as.numeric(scale(x)))



n_folds <- 10

set.seed(1)
folds    <- sample(rep(seq_len(n_folds), length.out = nrow(df)))
idx_list <- lapply(seq_len(n_folds), function(k) {
  list(tr = which(folds != k), te = which(folds == k))
})

basef <- as.formula("mean_RT ~ zone + position + wlen + wlen_so1 + wlen_so2 + unisurp + unisurp_so1 + unisurp_so2")

tgt2gf <- as.formula("mean_RT ~ zone + position + wlen + wlen_so1 + wlen_so2 + unisurp + unisurp_so1 + unisurp_so2 + gpt2_ctx2 + gpt2_ctx2_so1 + gpt2_ctx2_so2")
tgt100gf <- as.formula("mean_RT ~ zone + position + wlen + wlen_so1 + wlen_so2 + unisurp + unisurp_so1 + unisurp_so2 + gpt2_ctx100 + gpt2_ctx100_so1 + gpt2_ctx100_so2")

df$ll_base <- NA_real_
df$ll_2g   <- NA_real_
df$ll_100g <- NA_real_

for (k in seq_len(n_folds)) {
  tr <- df[idx_list[[k]]$tr, ]
  te <- df[idx_list[[k]]$te, ]
  te_idx <- idx_list[[k]]$te
  y  <- te$mean_RT

  m_base <- lm(basef, data = tr)
  df$ll_base[te_idx] <- dnorm(y, mean = predict(m_base, newdata = te), sd = sigma(m_base), log = TRUE)

  m_2g <- lm(tgt2gf, data = tr)
  df$ll_2g[te_idx] <- dnorm(y, mean = predict(m_2g, newdata = te), sd = sigma(m_2g), log = TRUE)

  m_100g <- lm(tgt100gf, data = tr)
  df$ll_100g[te_idx] <- dnorm(y, mean = predict(m_100g, newdata = te), sd = sigma(m_100g), log = TRUE)
}

df <- df %>%
  mutate(
    dll_2g = ll_2g - ll_base,
    dll_100g = ll_100g - ll_base,
    ddll = dll_100g - dll_2g
  )

df_filtered <- df %>%
  group_by(pos) %>%
  mutate(pos_freq = n()) %>%
  filter(pos_freq > 50) %>%
  ungroup() %>%
  mutate(scaled_ddll = as.numeric(scale(ddll)))

m_pos <- lm(scaled_ddll ~ pos - 1, data = df_filtered)

m_coef <- summary(m_pos)$coefficients
m_ci   <- confint(m_pos)

pos_results <- data.frame(
  term      = rownames(m_coef),
  estimate  = m_coef[, "Estimate"],
  conf.low  = m_ci[, 1],
  conf.high = m_ci[, 2],
  stringsAsFactors = FALSE
) %>%
  mutate(pos = str_remove(term, "^pos")) %>%
  arrange(desc(estimate)) %>%
  mutate(pos = factor(pos, levels = pos))

p_pos <- ggplot(pos_results, aes(x = estimate, y = pos)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "#63666A") +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2, color = "#041E42") +
  geom_point(shape = 1, size = 3, color = "#041E42", stroke = 1) +  
  labs(
    title = "Impact of Context Size (99 vs 1 tokens) by POS",
    x = "Delta Delta Log-Likelihood (Scaled)",
    y = "Part of Speech (POS)"
  ) +
  theme_bw() +
  theme(
    axis.text.y = element_text(size = 10),
    plot.title  = element_text(face = "bold")
  )
p_pos
ggsave("../result/fig/ns_pos_ddll.png", p_pos, width = 8, height = 8, dpi = 300)





library(brms)
m_pos_brm <- brm(
  formula = scaled_ddll ~ pos - 1,
  data = df_filtered,
  family = gaussian(),
  chains = 4,
  cores = parallel::detectCores(),
  iter = 2000,
  seed = 1
)
m_coef <- fixef(m_pos_brm)
pos_results <- data.frame(
  term      = rownames(m_coef),
  estimate  = m_coef[, "Estimate"],
  conf.low  = m_coef[, "Q2.5"],
  conf.high = m_coef[, "Q97.5"],
  stringsAsFactors = FALSE
) %>%
  mutate(pos = str_remove(term, "^pos")) %>%
  arrange(desc(estimate)) %>%
  mutate(pos = factor(pos, levels = pos))

p_pos <- ggplot(pos_results, aes(x = estimate, y = pos)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "#63666A") +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2, color = "#041E42") +
  geom_point(shape = 1, size = 3, color = "#041E42", stroke = 1) +  
  labs(
    title = "Impact of Context Size (99 vs 1 tokens) by POS (Bayesian)",
    x = "Delta Delta Log-Likelihood (Scaled)",
    y = "Part of Speech (POS)"
  ) +
  theme_bw() +
  theme(
    axis.text.y = element_text(size = 10),
    plot.title  = element_text(face = "bold")
  )

p_pos
ggsave("../result/fig/ns_pos_brms_ddll.png", p_pos, width = 8, height = 8, dpi = 300)
