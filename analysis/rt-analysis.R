library(readr)
library(dplyr)
library(tidyverse)
library(data.table)
library(doSNOW)
library(foreach)
library(ggplot2)

set.seed(1)

surp_base <- "../data/surp"
ns_surp_base <- file.path(surp_base, "ns")
brown_surp_base <- file.path(surp_base, "brown")
os_surp_base <- file.path(surp_base, "os")
provo_surp_base <- file.path(surp_base, "provo")

ns_model_dirs <- list.dirs(ns_surp_base, recursive = FALSE, full.names = FALSE)
brown_model_dirs <- list.dirs(brown_surp_base, recursive = FALSE, full.names = FALSE)
os_model_dirs <- list.dirs(os_surp_base, recursive = FALSE, full.names = FALSE)
provo_model_dirs <- list.dirs(provo_surp_base, recursive = FALSE, full.names = FALSE)

all_dirs_lists <- list(ns = ns_model_dirs, brown = brown_model_dirs, os = os_model_dirs, provo = provo_model_dirs)
identical_check <- sapply(all_dirs_lists, function(x) identical(sort(x), sort(all_dirs_lists[[1]])))
stopifnot("The directories contain different model names" = all(identical_check))
unique_counts <- unique(lengths(all_dirs_lists))
stopifnot("The directories contain different numbers of files" = length(unique_counts) == 1)

discover_contexts <- function(model_name) {
    files <- list.files(
        file.path(ns_surp_base, model_name),
        pattern = "^context_\\d+\\.txt$",
        full.names = FALSE
    )
    n_values <- as.integer(gsub("context_|\\.txt", "", files))
    df <- data.frame(
        file_name = files,
        n_gram = n_values,
        context_size = n_values - 1,
        stringsAsFactors = FALSE
    )
    return(df[order(df$context_size), ])
}

model_dirs <- all_dirs_lists[[1]]
contexts_per_model <- setNames(lapply(model_dirs, discover_contexts), model_dirs)
all_context_sizes <- sort(unique(unlist(lapply(contexts_per_model, function(df) df$context_size))))
message(sprintf("Found %d model(s): %s", length(model_dirs), paste(model_dirs, collapse = ", ")))
message(sprintf("Found %d unique context size(s): %s", length(all_context_sizes), paste(all_context_sizes, collapse = ", ")))

CANONICAL_ORDER <- c("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
model_order <- c(CANONICAL_ORDER[CANONICAL_ORDER %in% model_dirs], sort(setdiff(model_dirs, CANONICAL_ORDER)))

model_colors <- setNames(
    c("#FF4B00", "#005AFF", "#03AF7A", "#4DC4FF"),
    model_order
)
model_labels <- setNames(sub("^gpt2$", "gpt2-small", model_order), model_order)

corpora <- list(
    ns = list(surp_base = ns_surp_base, preds_path = "../data/baselines_ns.csv"),
    brown = list(surp_base = brown_surp_base, preds_path = "../data/brown_spr/preds.csv"),
    os = list(surp_base = os_surp_base, preds_path = "../data/OneStop/preds.csv"),
    provo = list(surp_base = provo_surp_base, preds_path = "../data/provo_corpus/preds.csv")
)

preds_list <- lapply(corpora, function(cfg) read_csv(cfg$preds_path, show_col_types = FALSE))

for (corp in names(corpora)) {
    for (m in model_dirs) {
        mc <- gsub("[-.]", "_", m)
        ctx_df <- contexts_per_model[[m]]
        for (i in seq_len(nrow(ctx_df))) {
        ng <- ctx_df$n_gram[i]
        fpath <- file.path(corpora[[corp]]$surp_base, m, sprintf("context_%d.txt", ng))
        if (!file.exists(fpath)) next
        vals <- scan(fpath, what = numeric(), quiet = TRUE)
        if (length(vals) != nrow(preds_list[[corp]])) {
            warning(sprintf("[%s] Row mismatch for %s ctx%d: expected %d, got %d — skipping", corp, m, ng, nrow(preds_list[[corp]]), length(vals)))
            next
        }
        preds_list[[corp]][[sprintf("%s_ctx%d", mc, ng)]] <- vals
        }
    }
}

message("Computing correlation with unigram surprisal...")
sanity_rows <- list()
for (corp in names(corpora)) {
    preds <- preds_list[[corp]]
    uni <- preds[["unisurp"]]
    for (m in model_dirs) {
        mc <- gsub("[-.]", "_", m)
        ctx_df <- contexts_per_model[[m]]
        for (i in seq_len(nrow(ctx_df))) {
        col <- sprintf("%s_ctx%d", mc, ctx_df$n_gram[i])
        if (!col %in% names(preds)) next
        sanity_rows[[length(sanity_rows) + 1]] <- data.frame(
            corpus = corp,
            model = m,
            context = ctx_df$context_size[i],
            cor_with_unisurp = cor(preds[[col]], uni, use = "complete.obs"),
            mean_surp = mean(preds[[col]], na.rm = TRUE),
            stringsAsFactors = FALSE
        )
        }
    }
}
sanity_df <- bind_rows(sanity_rows) %>%
    mutate(model = factor(model, levels = model_order), corpus = factor(corpus, levels = names(corpora)))
# write.csv(sanity_df, "./result/sanity_check.csv", row.names = FALSE)
# message("  Saved: ./result/sanity_check.csv")

make_sanity_plot <- function(df, y, ylab) {
    ggplot(df %>% mutate(context = factor(context, levels = sort(unique(context)))), aes(x = context, y = .data[[y]], color = model, group = model)) +
        geom_line(linewidth = 0.6, alpha = 0.7) +
        geom_point(size = 2) +
        facet_wrap(~ corpus, ncol = 2,
                   labeller = as_labeller(c(ns = "Natural Stories", brown = "Brown", os = "OneStop", provo = "Provo"))) +
        scale_x_discrete(name = "Context window (tokens)") +
        scale_color_manual(values = model_colors, labels = model_labels) +
        ylab(ylab) + labs(color = NULL) +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "bottom")
}

sanity_plots <- list(
    list(y = "cor_with_unisurp", ylab = "Pearson r with unigram surprisal", file = "../result/fig/unigram_cor.png"),
    list(y = "mean_surp", ylab = "Mean surprisal (bits)", file = "../result/fig/mean_surp.png")
)
for (plt in sanity_plots) {
    ggsave(plt$file, make_sanity_plot(sanity_df, plt$y, plt$ylab), width = 12, height = 8, dpi = 300)
    message(sprintf("Saved: %s", plt$file))
}


message("Building spillover terms...")

NO_SPILL <- c("story", "zone", "word", "bos", "eos", "is_punct", "position", "pos", "surp")

build_spillover <- function(df) {
    sp_cols <- setdiff(names(df), NO_SPILL)
    make_shift <- function(offset, sfx)
        df %>% mutate(zone = zone + offset) %>%
        select(story, zone, all_of(sp_cols)) %>%
        rename_with(~ paste0(., sfx), .cols = all_of(sp_cols))
    df %>%
        merge(make_shift(1, "_so1"), by = c("story", "zone"), sort = FALSE) %>%
        merge(make_shift(2, "_so2"), by = c("story", "zone"), sort = FALSE) %>%
        filter(is_punct == 0, bos == 0, eos == 0) %>%
        select(-is_punct, -bos, -eos) %>%
        drop_na()
}

scale_df <- function(df) {
    sc <- grep("^(zone|wlen|unisurp)|_ctx\\d+", names(df), value = TRUE, perl = TRUE)
    df[sc] <- lapply(df[sc], function(x) as.numeric(scale(x)))
    df
}

preds_list[["os"]] <- preds_list[["os"]] %>%
  mutate(story = paste(article_batch, article_id, difficulty_level, sep = "-")) %>%
  select(-article_batch, -article_id, -difficulty_level)

pred_proc <- lapply(preds_list, build_spillover)
for (corp in names(pred_proc)) message(sprintf("[%s] after filtering: %d rows", corp, nrow(pred_proc[[corp]])))



load_rt <- list(
    ns_spr = function()
        read.table("../data/naturalstories/naturalstories_RTS/processed_RTS.tsv", sep = "\t", quote = "", header = TRUE) %>%
        select(WorkerId, item, zone, RT) %>% rename(story = item) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(RT, na.rm = TRUE), .groups = "drop"),

    ns_maze = function()
        read_rds("../data/maze/maze.rds") %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(maze_RT, na.rm = TRUE), .groups = "drop"),

    brown = function()
        read_csv("../data/brown_spr/brown_spr.csv", col_types = cols_only(text_id = "d", text_pos = "d", time = "d")) %>%
        rename(story = text_id, zone = text_pos) %>%
        mutate(story = story + 1, zone = zone + 1) %>%
        filter(time > 100, time <= 3000) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(time, na.rm = TRUE), .groups = "drop"),

    osfp = function()
        read_csv("../data/OneStop/rts.csv", col_types = cols_only(article_batch = "c", article_id = "c", difficulty_level = "c", zone = "d", is_correct = "l", IA_FIRST_RUN_DWELL_TIME = "d"), na = c("", "NA", ".")) %>%
        mutate(story = paste(article_batch, article_id, difficulty_level, sep = "-")) %>%
        select(-article_batch, -article_id, -difficulty_level) %>%
        filter(is_correct == TRUE) %>%
        mutate(IA_FIRST_RUN_DWELL_TIME = replace_na(IA_FIRST_RUN_DWELL_TIME, 0)) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(IA_FIRST_RUN_DWELL_TIME, na.rm = TRUE), .groups = "drop"),

    osgp = function()
        read_csv("../data/OneStop/rts.csv", col_types = cols_only(article_batch = "c", article_id = "c", difficulty_level = "c", zone = "d", is_correct = "l", IA_REGRESSION_PATH_DURATION = "d"), na = c("", "NA", ".")) %>%
        mutate(story = paste(article_batch, article_id, difficulty_level, sep = "-")) %>%
        select(-article_batch, -article_id, -difficulty_level) %>%
        filter(is_correct == TRUE) %>%
        mutate(IA_REGRESSION_PATH_DURATION = replace_na(IA_REGRESSION_PATH_DURATION, 0)) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(IA_REGRESSION_PATH_DURATION, na.rm = TRUE), .groups = "drop"),

    ost = function()
        read_csv("../data/OneStop/rts.csv", col_types = cols_only(article_batch = "c", article_id = "c", difficulty_level = "c", zone = "d", is_correct = "l", IA_DWELL_TIME = "d"), na = c("", "NA", ".")) %>%
        mutate(story = paste(article_batch, article_id, difficulty_level, sep = "-")) %>%
        select(-article_batch, -article_id, -difficulty_level) %>%
        filter(is_correct == TRUE) %>%
        mutate(IA_DWELL_TIME = replace_na(IA_DWELL_TIME, 0)) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(IA_DWELL_TIME, na.rm = TRUE), .groups = "drop"),

    provofp = function()
        read_csv("../data/provo_corpus/Provo_Corpus-Eyetracking_Data.csv", col_types = cols_only(Participant_ID = "c", Text_ID = "d", Word_Number = "d", IA_FIRST_RUN_DWELL_TIME = "d")) %>%
        rename(story = Text_ID, zone = Word_Number) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(IA_FIRST_RUN_DWELL_TIME, na.rm = TRUE), .groups = "drop"),

    provogp = function()
        read_csv("../data/provo_corpus/Provo_Corpus-Eyetracking_Data.csv", col_types = cols_only(Participant_ID = "c", Text_ID = "d", Word_Number = "d", IA_REGRESSION_PATH_DURATION = "d")) %>%
        rename(story = Text_ID, zone = Word_Number) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(IA_REGRESSION_PATH_DURATION, na.rm = TRUE), .groups = "drop"),

    provot = function()
        read_csv("../data/provo_corpus/Provo_Corpus-Eyetracking_Data.csv", col_types = cols_only(Participant_ID = "c", Text_ID = "d", Word_Number = "d", IA_DWELL_TIME = "d")) %>%
        rename(story = Text_ID, zone = Word_Number) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(IA_DWELL_TIME, na.rm = TRUE), .groups = "drop")
)

ds_config <- list(
    ns_spr = list(corpus = "ns", has_pos = TRUE),
    ns_maze = list(corpus = "ns", has_pos = TRUE),
    brown = list(corpus = "brown", has_pos = FALSE),
    osfp = list(corpus = "os", has_pos = TRUE),
    osgp = list(corpus = "os", has_pos = TRUE),
    ost = list(corpus = "os", has_pos = TRUE),
    provofp = list(corpus = "provo", has_pos = FALSE),
    provogp = list(corpus = "provo", has_pos = FALSE),
    provot = list(corpus = "provo", has_pos = FALSE)
)

df_list <- lapply(names(ds_config), function(ds) {
    cfg <- ds_config[[ds]]
    rt  <- load_rt[[ds]]()
    scale_df(merge(rt, pred_proc[[cfg$corpus]], by = c("story", "zone"), sort = FALSE) %>% ungroup() %>% drop_na())
})
names(df_list) <- names(ds_config)
for (ds in names(df_list)) message(sprintf("[%s] after RT merge: %d rows", ds, nrow(df_list[[ds]])))


n_folds <- 10
n_cores <- max(1L, parallel::detectCores() - 2L)
message(sprintf("Running %d-fold CV using %d cores...", n_folds, n_cores))

all_results <- list()

for (ds_name in names(ds_config)) {
    message(sprintf("Dataset: %s", ds_name))
    df <- df_list[[ds_name]]
    has_pos <- ds_config[[ds_name]]$has_pos
    corpus <- ds_config[[ds_name]]$corpus
    outcome <- "mean_RT"

    base_f <- function(extra) as.formula(sprintf("%s ~ zone + wlen + wlen_so1 + wlen_so2%s", outcome, extra))

    set.seed(1)
    folds <- sample(rep(seq_len(n_folds), length.out = nrow(df)))
    idx_list <- lapply(seq_len(n_folds), function(k) list(tr = which(folds != k), te = which(folds == k)))

    cache_base <- function(f) lapply(seq_len(n_folds), function(k) {
        m <- lm(f, data = df[idx_list[[k]]$tr, ])
        list(pred  = predict(m, newdata = df[idx_list[[k]]$te, ]),
            sigma = sigma(m), y = df[[outcome]][idx_list[[k]]$te])
    })
    base1_cache <- cache_base(base_f(""))
    base2_cache <- cache_base(base_f("+ unisurp + unisurp_so1 + unisurp_so2"))

    param_grid <- do.call(rbind, lapply(model_dirs, function(m) {
        mc <- gsub("[-.]", "_", m)
        ctx_df <- contexts_per_model[[m]]
        valid <- sapply(seq_len(nrow(ctx_df)), function(i) {
        ng <- ctx_df$n_gram[i]
        all(sprintf("%s_ctx%d%s", mc, ng, c("", "_so1", "_so2")) %in% names(df))
        })
        if (!any(valid)) return(NULL)
        data.frame(model = m, n_gram = ctx_df$n_gram[valid], context = ctx_df$context_size[valid], stringsAsFactors = FALSE)
    }))
    message(sprintf("%d (model x context) combinations", nrow(param_grid)))

    cl <- parallel::makeCluster(n_cores)
    doSNOW::registerDoSNOW(cl)
    pb <- txtProgressBar(min = 0, max = nrow(param_grid), style = 3)
    opts <- list(progress = function(i) setTxtProgressBar(pb, i))

    res <- foreach::foreach(
        i             = seq_len(nrow(param_grid)),
        .combine      = rbind,
        .packages     = "stats",
        .export       = c("df", "idx_list", "base1_cache", "base2_cache", "n_folds", "outcome"),
        .options.snow = opts
    ) %dopar% {
        pg    <- param_grid[i, ]
        mname <- pg$model
        mc    <- gsub("[-.]", "_", mname)
        ng    <- as.integer(pg$n_gram)
        ctx   <- as.integer(pg$context)

        make_f <- function(extra) as.formula(sprintf(
        "%s ~ zone + wlen + wlen_so1 + wlen_so2%s + %s_ctx%d + %s_ctx%d_so1 + %s_ctx%d_so2",
        outcome, extra, mc, ng, mc, ng, mc, ng))
        tgt1_f <- make_f("")
        tgt2_f <- make_f("+ unisurp + unisurp_so1 + unisurp_so2")

        dll_fold <- function(k, cache, tgt_f) {
        y <- cache[[k]]$y
        m_t <- lm(tgt_f, data = df[idx_list[[k]]$tr, ])
        ll_t <- dnorm(y, predict(m_t, df[idx_list[[k]]$te, ]), sigma(m_t), log = TRUE)
        ll_b <- dnorm(y, cache[[k]]$pred, cache[[k]]$sigma, log = TRUE)
        mean(ll_t - ll_b, na.rm = TRUE)
        }
        smry <- function(vals) {
        n <- sum(!is.na(vals)); mn <- mean(vals, na.rm = TRUE)
        se <- if (n > 1) sd(vals, na.rm = TRUE) / sqrt(n) else NA_real_
        c(mean = mn, lower = mn - 1.96 * se, upper = mn + 1.96 * se)
        }
        r1 <- smry(sapply(seq_len(n_folds), dll_fold, base1_cache, tgt1_f))
        r2 <- smry(sapply(seq_len(n_folds), dll_fold, base2_cache, tgt2_f))
        data.frame(model = mname, context = ctx, n_folds = n_folds,
                mean_dll_no_unisurp   = r1["mean"], lower_ci_no_unisurp   = r1["lower"], upper_ci_no_unisurp   = r1["upper"],
                mean_dll_with_unisurp = r2["mean"], lower_ci_with_unisurp = r2["lower"], upper_ci_with_unisurp = r2["upper"],
                stringsAsFactors = FALSE)
    }
    close(pb)
    parallel::stopCluster(cl)
    rownames(res) <- NULL
    all_results[[ds_name]] <- res %>% mutate(dataset = ds_name, corpus = corpus)
}

results <- bind_rows(all_results) %>% mutate(model = factor(model, levels = model_order))
write.csv(results, "../result/lm_dll.csv", row.names = FALSE)
message("Saved: ../result/lm_dll.csv")
results <- read.csv("../result/lm_dll.csv")


context_order <- sort(unique(results$context))

plot_long <- results %>%
    mutate(context = factor(context, levels = context_order),
            model = factor(model, levels = model_order),
            dataset = factor(dataset, levels = names(ds_config))) %>%
    pivot_longer(c(mean_dll_no_unisurp, mean_dll_with_unisurp),
                names_to = "baseline_type", values_to = "mean_dll") %>%
    mutate(
        lower_ci = if_else(baseline_type == "mean_dll_no_unisurp", lower_ci_no_unisurp, lower_ci_with_unisurp),
        upper_ci = if_else(baseline_type == "mean_dll_no_unisurp", upper_ci_no_unisurp, upper_ci_with_unisurp),
        baseline_label = if_else(baseline_type == "mean_dll_no_unisurp", "w/o unigram surp", "w/ unigram surp")
    )

combined_long <- plot_long %>%
    mutate(context_num = as.integer(as.character(context))) %>%
    left_join(
        sanity_df %>%
        mutate(context_num = as.integer(context), corpus = as.character(corpus)) %>%
        select(corpus, model, context_num, cor_with_unisurp, mean_surp),
        by = c("corpus", "model", "context_num")
    ) %>%
    select(-context_num)

ds_labels <- c(
    ns_spr  = "Natural Stories SPR",
    ns_maze = "Natural Stories A-Maze",
    brown   = "Brown SPR",
    osfp    = "OneStop FPD",
    osgp    = "OneStop GPD",
    ost     = "OneStop TFD",
    provofp = "Provo FPD",
    provogp = "Provo GPD",
    provot  = "Provo TFD"
)

make_plot <- function(df) {
    ggplot(df, aes(x = context, y = mean_dll, color = model, group = model)) +
        geom_line(linewidth = 0.6, alpha = 0.7) +
        geom_point(size = 1.5) +
        geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), width = 0.3, linewidth = 0.4, alpha = 0.5) +
        facet_wrap(~ dataset_label, nrow = 3, scales = "free_y") +
        scale_x_discrete(name = "Context window (tokens)") +
        scale_color_manual(values = model_colors, labels = model_labels) +
        ylab("Delta Log Likelihood (per word)") +
        labs(color = NULL) +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1),
              strip.text = element_text(face = "bold"), legend.position = "bottom")
}

p_files <- list(
    "w/o unigram surp" = "../result/fig/lm_dll_no_unisurp.png",
    "w/ unigram surp"  = "../result/fig/lm_dll_with_unisurp.png"
)
for (bl in names(p_files)) {
    sub <- plot_long %>%
        filter(baseline_label == bl) %>%
        mutate(dataset_label = factor(ds_labels[as.character(dataset)], levels = unname(ds_labels)))
    ggsave(p_files[[bl]], make_plot(sub), width = 12, height = 6, dpi = 300)
    message(sprintf("Saved: %s", p_files[[bl]]))
}
