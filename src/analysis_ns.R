library(readr)
library(dplyr)
library(tidyverse)
library(data.table)
library(doSNOW)
library(foreach)
library(ggplot2)

set.seed(1)

ns_surp_base <- "./data/ns_surp"


message("Discovering models and context sizes from ", ns_surp_base, " ...")
model_dirs <- list.dirs(ns_surp_base, recursive = FALSE, full.names = FALSE)
if (length(model_dirs) == 0) stop("No model directories found under ", ns_surp_base)

# For each model directory, list available context sizes
discover_contexts <- function(model_name) {
  files <- list.files(
    file.path(ns_surp_base, model_name),
    pattern = "^context_\\d+\\.txt$",
    full.names = FALSE
  )
  sort(as.integer(gsub("context_|\\.txt", "", files)))
}

contexts_per_model <- lapply(model_dirs, discover_contexts)
names(contexts_per_model) <- model_dirs

all_contexts <- sort(unique(unlist(contexts_per_model)))
message(sprintf("  Found %d model(s): %s", length(model_dirs), paste(model_dirs, collapse = ", ")))
message(sprintf("  Found %d unique context size(s): %s", length(all_contexts), paste(all_contexts, collapse = ", ")))

CANONICAL_ORDER <- c(
  "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
  "gpt-neo-125m", "gpt-neo-1.3b", "gpt-neo-2.7b", "gpt-j-6b",
  "opt-125m", "opt-350m", "opt-1.3b", "opt-2.7b", "opt-6.7b",
  "opt-13b", "opt-30b", "opt-66b",
  "pythia-14m", "pythia-31m", "pythia-70m", "pythia-160m",
  "pythia-410m", "pythia-1b", "pythia-1.4b", "pythia-2.8b",
  "pythia-6.9b", "pythia-12b"
)
model_order <- c(
  CANONICAL_ORDER[CANONICAL_ORDER %in% model_dirs],
  sort(setdiff(model_dirs, CANONICAL_ORDER))
)

get_model_family <- function(name) {
  if (grepl("^gpt2",          name)) return("gpt2")
  if (grepl("^gpt-neo|^gpt-j", name)) return("gpt-neo")
  if (grepl("^opt",            name)) return("opt")
  if (grepl("^pythia",         name)) return("pythia")
  return("other")
}

family_palettes <- list(
  "gpt2"    = colorRampPalette(c("#BDD7EE", "#1F497D")),  # blues
  "gpt-neo" = colorRampPalette(c("#B8E4B0", "#1A6B1A")),  # greens
  "opt"     = colorRampPalette(c("#FDDCB5", "#B35A00")),  # oranges
  "pythia"  = colorRampPalette(c("#D9B8E4", "#5A007A")),  # purples
  "other"   = colorRampPalette(c("#CCCCCC", "#444444"))
)

model_colors <- {
  fams   <- sapply(model_order, get_model_family)
  colors <- character(length(model_order))
  for (fam in unique(fams)) {
    idx <- which(fams == fam)
    colors[idx] <- family_palettes[[fam]](length(idx))
  }
  setNames(colors, model_order)
}

message("Loading baseline predictors...")
baseline <- read_csv("./data/baselines_ns.csv", show_col_types = FALSE)
n_words <- nrow(baseline)
message(sprintf("  Baseline: %d rows", n_words))

message("Loading surprisal values...")
loaded <- 0
for (m in model_dirs) {
  mc <- gsub("[-.]", "_", m)
  for (ctx in contexts_per_model[[m]]) {
    fpath <- sprintf("%s/%s/context_%d.txt", ns_surp_base, m, ctx)
    vals  <- scan(fpath, what = numeric(), quiet = TRUE)
    if (length(vals) != n_words) {
      warning(sprintf("Row mismatch for %s ctx%d: expected %d, got %d — skipping", m, ctx, n_words, length(vals)))
      next
    }
    baseline[[sprintf("%s_ctx%d", mc, ctx)]] <- vals
    loaded <- loaded + 1
  }
}
message(sprintf("  Loaded %d surprisal columns.", loaded))

message("Computing sanity checks (correlation with unigram surprisal)...")

sanity_rows <- list()
for (m in model_dirs) {
  mc <- gsub("[-.]", "_", m)
  for (ctx in contexts_per_model[[m]]) {
    col <- sprintf("%s_ctx%d", mc, ctx)
    if (!col %in% names(baseline)) next
    vals <- baseline[[col]]
    uni  <- baseline[["unisurp"]]
    sanity_rows[[length(sanity_rows) + 1]] <- data.frame(
      model            = m,
      context          = ctx,
      cor_with_unisurp = cor(vals, uni, use = "complete.obs"),
      mean_surp        = mean(vals, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }
}
sanity_df <- bind_rows(sanity_rows) %>%
  mutate(model = factor(model, levels = model_order))
write.csv(sanity_df, "./result/ns_sanity_check.csv", row.names = FALSE)
message("  Saved: ./result/ns_sanity_check.csv")

p_cor <- ggplot(
  sanity_df %>% mutate(context = factor(context, levels = sort(unique(context)))),
  aes(x = context, y = cor_with_unisurp, color = model, group = model)
) +
  geom_line(linewidth = 0.6, alpha = 0.7) +
  geom_point(size = 2) +
  scale_x_discrete(name = "Context window (tokens)") +
  scale_color_manual(values = model_colors) +
  ylab("Pearson r with unigram surprisal") +
  labs(color = "Model") +
  theme_bw() +
  theme(
    axis.text.x     = element_text(angle = 45, hjust = 1),
    legend.position = "bottom"
  )
ggsave("./result/fig/ns_unigram_cor.png", p_cor, width = 10, height = 6, dpi = 300)
message("  Saved: ./result/fig/ns_unigram_cor.png")


p_mean_surp <- ggplot(
  sanity_df %>% mutate(context = factor(context, levels = sort(unique(context)))),
  aes(x = context, y = mean_surp, color = model, group = model)
) +
  geom_line(linewidth = 0.6, alpha = 0.7) +
  geom_point(size = 2) +
  scale_x_discrete(name = "Context window (tokens)") +
  scale_color_manual(values = model_colors) +
  ylab("Mean surprisal (bits)") +
  labs(color = "Model") +
  theme_bw() +
  theme(
    axis.text.x     = element_text(angle = 45, hjust = 1),
    legend.position = "bottom"
  )
ggsave("./result/fig/ns_mean_surp.png", p_mean_surp, width = 10, height = 6, dpi = 300)
message("  Saved: ./result/fig/ns_mean_surp.png")


message("Building spillover terms...")
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
message(sprintf("  After filtering bos/eos/punct: %d words", nrow(pred)))


message("Loading RT data...")
rt_path <- "./data/naturalstories/naturalstories_RTS/processed_RTs.tsv"
rts_summary <- read.table(rt_path, sep = "\t", quote = "", header = TRUE) %>%
  select(WorkerId, item, zone, RT) %>%
  rename(story = item) %>%
  group_by(story, zone) %>%
  summarise(mean_RT = mean(RT, na.rm = TRUE), .groups = "drop")

df <- merge(rts_summary, pred, by = c("story", "zone"), sort = FALSE)
message(sprintf("  After RT merge: %d rows", nrow(df)))

scale_cols <- grep("^(zone|position|wlen|unisurp)|_ctx\\d+", names(df), value = TRUE, perl = TRUE)
df[scale_cols] <- lapply(df[scale_cols], function(x) as.numeric(scale(x)))


n_folds <- 10
n_cores <- max(1L, parallel::detectCores() - 2L)
message(sprintf("Setting up %d-fold CV using %d cores...", n_folds, n_cores))

set.seed(1)
folds    <- sample(rep(seq_len(n_folds), length.out = nrow(df)))
idx_list <- lapply(seq_len(n_folds), function(k) {
  list(tr = which(folds != k), te = which(folds == k))
})


# zone: word position in the document
# position: word position in the sentence
base1_f <- as.formula("mean_RT ~ zone + position + wlen + wlen_so1 + wlen_so2")
base2_f <- as.formula("mean_RT ~ zone + position + wlen + wlen_so1 + wlen_so2 + unisurp + unisurp_so1 + unisurp_so2")

base1_cache <- vector("list", n_folds)
base2_cache <- vector("list", n_folds)
for (k in seq_len(n_folds)) {
  tr <- df[idx_list[[k]]$tr, ]
  te <- df[idx_list[[k]]$te, ]
  m1 <- lm(base1_f, data = tr)
  base1_cache[[k]] <- list(pred = predict(m1, newdata = te), sigma = sigma(m1), y = te$mean_RT)
  m2 <- lm(base2_f, data = tr)
  base2_cache[[k]] <- list(pred = predict(m2, newdata = te), sigma = sigma(m2), y = te$mean_RT)
}

param_grid <- do.call(rbind, lapply(model_dirs, function(m) {
  ctxs <- contexts_per_model[[m]]
  mc   <- gsub("[-.]", "_", m)
  valid_ctxs <- ctxs[sapply(ctxs, function(ctx) {
    all(c(
      sprintf("%s_ctx%d",     mc, ctx),
      sprintf("%s_ctx%d_so1", mc, ctx),
      sprintf("%s_ctx%d_so2", mc, ctx)
    ) %in% names(df))
  })]
  if (length(valid_ctxs) == 0) return(NULL)
  data.frame(model = m, context = valid_ctxs, stringsAsFactors = FALSE)
}))
message(sprintf("Parameter grid: %d (model × context) combinations", nrow(param_grid)))

# ---- Parallel 10-fold CV ----
cl <- parallel::makeCluster(n_cores)
on.exit(try(parallel::stopCluster(cl), silent = TRUE), add = TRUE)
doSNOW::registerDoSNOW(cl)

pb   <- txtProgressBar(min = 0, max = nrow(param_grid), style = 3)
opts <- list(progress = function(i) setTxtProgressBar(pb, i))

results <- foreach::foreach(
  i             = seq_len(nrow(param_grid)),
  .combine      = rbind,
  .packages     = c("stats"),
  .export       = c("df", "idx_list", "base1_cache", "base2_cache", "n_folds"),
  .options.snow = opts
) %dopar% {

  pg  <- param_grid[i, ]
  m   <- pg$model
  ctx <- as.integer(pg$context)
  mc  <- gsub("[-.]", "_", m)

  surp_col  <- sprintf("%s_ctx%d",     mc, ctx)
  surp1_col <- sprintf("%s_ctx%d_so1", mc, ctx)
  surp2_col <- sprintf("%s_ctx%d_so2", mc, ctx)

  tgt1_f <- as.formula(paste(
    "mean_RT ~ zone + position +",
    paste(c("wlen", "wlen_so1", "wlen_so2",
            surp_col, surp1_col, surp2_col), collapse = " + ")
  ))
  tgt2_f <- as.formula(paste(
    "mean_RT ~ zone + position +",
    paste(c("wlen", "wlen_so1", "wlen_so2",
            "unisurp", "unisurp_so1", "unisurp_so2",
            surp_col, surp1_col, surp2_col), collapse = " + ")
  ))

  dll1 <- numeric(n_folds)
  dll2 <- numeric(n_folds)

  for (k in seq_len(n_folds)) {
    tr <- df[idx_list[[k]]$tr, ]
    te <- df[idx_list[[k]]$te, ]
    y  <- base1_cache[[k]]$y

    m_t1  <- lm(tgt1_f, data = tr)
    ll_t1 <- dnorm(y, mean = predict(m_t1, newdata = te), sd = sigma(m_t1), log = TRUE)
    ll_b1 <- dnorm(y, mean = base1_cache[[k]]$pred, sd = base1_cache[[k]]$sigma, log = TRUE)
    dll1[k] <- mean(ll_t1 - ll_b1, na.rm = TRUE)

    m_t2  <- lm(tgt2_f, data = tr)
    ll_t2 <- dnorm(y, mean = predict(m_t2, newdata = te), sd = sigma(m_t2), log = TRUE)
    ll_b2 <- dnorm(y, mean = base2_cache[[k]]$pred, sd = base2_cache[[k]]$sigma, log = TRUE)
    dll2[k] <- mean(ll_t2 - ll_b2, na.rm = TRUE)
  }

  fold_summary <- function(vals) {
    n  <- sum(!is.na(vals))
    mn <- mean(vals, na.rm = TRUE)
    se <- if (n > 1) sd(vals, na.rm = TRUE) / sqrt(n) else NA_real_
    c(mean = mn, lower = mn - 1.96 * se, upper = mn + 1.96 * se)
  }
  r1 <- fold_summary(dll1)
  r2 <- fold_summary(dll2)

  data.frame(
    model   = m, context = ctx, n_folds = n_folds,
    mean_dll_no_unisurp   = r1["mean"],
    lower_ci_no_unisurp   = r1["lower"],
    upper_ci_no_unisurp   = r1["upper"],
    mean_dll_with_unisurp = r2["mean"],
    lower_ci_with_unisurp = r2["lower"],
    upper_ci_with_unisurp = r2["upper"],
    stringsAsFactors = FALSE
  )
}

close(pb)
try(parallel::stopCluster(cl), silent = TRUE)
rownames(results) <- NULL

write.csv(results, "./result/ns_lm_dll.csv", row.names = FALSE)
message("Results saved to ./result/ns_lm_dll.csv")



context_order <- sort(unique(results$context))

plot_long <- results %>%
  mutate(
    context = factor(context, levels = context_order),
    model   = factor(model, levels = model_order)
  ) %>%
  pivot_longer(
    cols      = c(mean_dll_no_unisurp, mean_dll_with_unisurp),
    names_to  = "baseline_type",
    values_to = "mean_dll"
  ) %>%
  mutate(
    lower_ci = if_else(baseline_type == "mean_dll_no_unisurp", lower_ci_no_unisurp, lower_ci_with_unisurp),
    upper_ci = if_else(baseline_type == "mean_dll_no_unisurp", upper_ci_no_unisurp, upper_ci_with_unisurp),
    baseline_label = case_when(
      baseline_type == "mean_dll_no_unisurp" ~ "Baseline: zone + position + wlen  (w/o unigram surp)",
      baseline_type == "mean_dll_with_unisurp" ~ "Baseline: zone + position + wlen + unigram surp"
    )
  )

p1 <- ggplot(plot_long, aes(x = context, y = mean_dll, color = model, group = model)) +
  geom_line(linewidth = 0.6, alpha = 0.7) +
  geom_point(size = 1.5) +
  geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), width = 0.3, linewidth = 0.4, alpha = 0.5) +
  facet_wrap(~ baseline_label, scales = "free_y", ncol = 1) +
  scale_x_discrete(name = "Context window (tokens)") +
  scale_color_manual(values = model_colors) +
  ylab("Delta Log Likelihood (per word)") +
  labs(color = "Model") +
  theme_bw() +
  theme(
    axis.text.x     = element_text(angle = 45, hjust = 1),
    strip.text      = element_text(face = "bold"),
    legend.position = "bottom"
  )
ggsave("./result/fig/ns_lm_dll_by_context.png", p1, width = 12, height = 9, dpi = 300)
message("  Saved: ./result/fig/ns_lm_dll_by_context.png")

p2 <- ggplot(plot_long, aes(x = context, y = mean_dll, color = baseline_label, group = baseline_label)) +
  geom_line(linewidth = 0.6, alpha = 0.7) +
  geom_point(size = 1.5) +
  geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci),
                width = 0.3, linewidth = 0.4, alpha = 0.5) +
  facet_wrap(~ model, scales = "free_y") +
  scale_x_discrete(name = "Context window (tokens)") +
  ylab("Delta Log Likelihood (per word)") +
  labs(color = "Baseline") +
  theme_bw() +
  theme(
    axis.text.x      = element_text(angle = 45, hjust = 1),
    strip.text       = element_text(face = "bold"),
    legend.position  = "bottom",
    legend.direction = "vertical"
  )
ggsave("./result/fig/ns_lm_dll_by_model.png", p2, width = 14, height = 10, dpi = 300)
message("  Saved: ./result/fig/ns_lm_dll_by_model.png")


p3 <- ggplot(plot_long, aes(x = context, y = mean_dll, color = model, group = model)) +
  geom_line(linewidth = 0.5, alpha = 0.7) +
  geom_point(size = 1.2) +
  geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci),
                width = 0.25, linewidth = 0.4, alpha = 0.5) +
  facet_grid(rows = vars(baseline_label), cols = vars(model), scales = "free_y") +
  scale_x_discrete(name = "Context window (tokens)") +
  scale_color_manual(values = model_colors) +
  ylab("Delta Log Likelihood (per word)") +
  theme_bw() +
  theme(
    axis.text.x     = element_text(angle = 45, hjust = 1),
    strip.text      = element_text(face = "bold"),
    legend.position = "none"
  )
ggsave("./result/fig/ns_lm_dll_per_model.png", p3, width = 14, height = 8, dpi = 300)
message("  Saved: ./result/fig/ns_lm_dll_per_model.png")


combined_long <- plot_long %>%
  left_join(
    sanity_df %>% mutate(context = factor(context, levels = context_order)),
    by = c("model", "context")
  )

p4 <- ggplot(
  sanity_df %>%
    mutate(context = factor(context, levels = sort(unique(context)))),
  aes(x = context, y = mean_surp, color = model, group = model)
) +
  geom_line(linewidth = 0.6, alpha = 0.7) +
  geom_point(size = 1.5) +
  scale_x_discrete(name = "Context window (tokens)") +
  scale_color_manual(values = model_colors) +
  ylab("Mean surprisal (bits per word)") +
  labs(color = "Model") +
  theme_bw() +
  theme(
    axis.text.x     = element_text(angle = 45, hjust = 1),
    legend.position = "bottom"
  )
ggsave("./result/fig/ns_lm_mean_surp.png", p4, width = 10, height = 6, dpi = 300)
message("  Saved: ./result/fig/ns_lm_mean_surp.png")

p5 <- ggplot(combined_long, aes(x = mean_surp, y = mean_dll, color = model)) +
  geom_point(size = 1.2) +
  geom_text(aes(label = context), color = "black", size = 1.8, vjust = -0.4) +
  facet_wrap(~ baseline_label, scales = "free", ncol = 1) +
  scale_x_continuous(name = "Mean surprisal (bits per word)") +
  scale_color_manual(values = model_colors) +
  ylab("Delta Log Likelihood (per word)") +
  labs(color = "Model") +
  theme_bw() +
  theme(
    strip.text      = element_text(face = "bold"),
    legend.position = "bottom"
  )
ggsave("./result/fig/ns_lm_dll_vs_mean_surp.png", p5, width = 10, height = 10, dpi = 300)
message("  Saved: ./result/fig/ns_lm_dll_vs_mean_surp.png")


# Use only a subset of context sizes to keep the plot readable
ctx_subset <- with(combined_long, {
  all_ctx <- sort(unique(as.integer(as.character(context))))
  idx <- round(seq(1, length(all_ctx), length.out = min(16, length(all_ctx))))
  all_ctx[idx]
})

p6 <- ggplot(
  combined_long %>%
    filter(as.integer(as.character(context)) %in% ctx_subset),
  aes(x = mean_surp, y = mean_dll, color = model)
) +
  geom_point(size = 1.5) +
  geom_smooth(
    aes(group = 1),  # single regression across all models per facet
    method = "lm", formula = y ~ x, se = TRUE,
    color = "red", linewidth = 0.4, alpha = 0.2
  ) +
  facet_grid(rows = vars(baseline_label), cols = vars(context), scales = "free") +
  scale_x_continuous(name = "Mean surprisal (bits per word)") +
  scale_color_manual(values = model_colors) +
  ylab("Delta Log Likelihood (per word)") +
  labs(color = "Model",
       title = "DLL vs mean surprisal, by context window\n(Natural Stories SPR)") +
  theme_bw() +
  theme(
    axis.text.x     = element_text(angle = 45, hjust = 1),
    strip.text      = element_text(face = "bold"),
    legend.position = "bottom"
  )
ggsave("./result/fig/ns_lm_dll_vs_mean_surp_by_ctx.png", p6, width = max(16, length(ctx_subset) * 1.8), height = 9, dpi = 300)
message("  Saved: ./result/fig/ns_lm_dll_vs_mean_surp_by_ctx.png")


p7 <- ggplot(combined_long, aes(x = cor_with_unisurp, y = mean_dll, color = model)) +
  geom_point(size = 1.2) +
  geom_path(aes(group = model), linewidth = 0.4, alpha = 0.5) +
  geom_text(aes(label = context), color = "black", size = 1.8, vjust = -0.4) +
  facet_wrap(~ baseline_label, scales = "free_y", ncol = 1) +
  scale_x_continuous(
    name = "Pearson r(context-limited surp, unigram surp)\n[shorter context \u2192 larger r]"
  ) +
  scale_color_manual(values = model_colors) +
  ylab("Delta Log Likelihood (per word)") +
  labs(color = "Model") +
  theme_bw() +
  theme(
    strip.text      = element_text(face = "bold"),
    legend.position = "bottom"
  )
ggsave("./result/fig/ns_lm_dll_vs_cor_unisurp.png", p7, width = 10, height = 10, dpi = 300)
message("  Saved: ./result/fig/ns_lm_dll_vs_cor_unisurp.png")

message("All done!")
