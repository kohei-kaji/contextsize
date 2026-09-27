library(readr)
library(dplyr)
library(tidyr)
library(doSNOW)
library(foreach)

set.seed(12345)

PROVO_MAX_CTX <- 100L

surp_base    <- "../data/surp"
ns_surp_base <- file.path(surp_base, "ns")

ns_model_dirs <- list.dirs(ns_surp_base, recursive = FALSE, full.names = FALSE)
CANONICAL_ORDER <- c("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
model_order <- c(CANONICAL_ORDER[CANONICAL_ORDER %in% ns_model_dirs], sort(setdiff(ns_model_dirs, CANONICAL_ORDER)))

discover_contexts <- function(model_name) {
    files  <- list.files(file.path(ns_surp_base, model_name), pattern = "^context_\\d+\\.txt$", full.names = FALSE)
    n_vals <- as.integer(gsub("context_|\\.txt", "", files))
    data.frame(file_name = files, n_gram = n_vals, context_size = n_vals - 1L, stringsAsFactors = FALSE)[order(n_vals), ]
}
contexts_per_model <- setNames(lapply(model_order, discover_contexts), model_order)
message(sprintf("Found %d model(s): %s", length(model_order), paste(model_order, collapse = ", ")))
message(sprintf("Found %d unique context size(s): %s", length(sort(unique(unlist(lapply(contexts_per_model, `[[`, "context_size"))))), paste(sort(unique(unlist(lapply(contexts_per_model, `[[`, "context_size")))), collapse = ", ")))

# Corpus config
# has_pron: whether pronominalized surp files exist for this corpus
corpus_cfg <- list(
    ns = list(
        surp_dir    = file.path(surp_base, "ns"),
        preds       = "../data/baselines_ns.csv",
        pron_dir    = file.path(surp_base, "ns_pronominalized"),
        pron_bl_dir = file.path(surp_base, "ns_baseline_pronominalized"),
        has_pron    = TRUE
    ),
    brown = list(
        surp_dir    = file.path(surp_base, "brown"),
        preds       = "../data/brown_spr/preds.csv",
        pron_dir    = file.path(surp_base, "brown_pronominalized"),
        pron_bl_dir = file.path(surp_base, "brown_baseline_pronominalized"),
        has_pron    = TRUE
    ),
    os = list(
        surp_dir    = file.path(surp_base, "os"),
        preds       = "../data/OneStop/preds.csv",
        pron_dir    = file.path(surp_base, "onestop_pronominalized"),
        pron_bl_dir = file.path(surp_base, "onestop_baseline_pronominalized"),
        has_pron    = TRUE
    ),
    provo = list(
        surp_dir    = file.path(surp_base, "provo"),
        preds       = "../data/provo_corpus/preds.csv",
        pron_dir    = file.path(surp_base, "provo_pronominalized"),
        pron_bl_dir = file.path(surp_base, "provo_baseline_pronominalized"),
        has_pron    = TRUE
    )
)

# Load preds and attach surprisal columns
load_surp_safe <- function(fpath, n) {
    if (!file.exists(fpath)) return(NULL)
    v <- scan(fpath, what = numeric(), quiet = TRUE)
    if (length(v) != n) {
        warning(sprintf("Row mismatch %s: expected %d, got %d — skipped", fpath, n, length(v)))
        return(NULL)
    }
    v
}

message("Loading preds and surp files...")
preds_list <- lapply(corpus_cfg, function(cfg) read_csv(cfg$preds, show_col_types = FALSE))

for (corp in names(corpus_cfg)) {
    cfg <- corpus_cfg[[corp]]
    n   <- nrow(preds_list[[corp]])
    for (m in model_order) {
        mc     <- gsub("[-.]", "_", m)
        ctx_df <- contexts_per_model[[m]]
        for (i in seq_len(nrow(ctx_df))) {
            ng <- ctx_df$n_gram[i]
            v <- load_surp_safe(file.path(cfg$surp_dir, m, sprintf("context_%d.txt", ng)), n)
            if (!is.null(v)) preds_list[[corp]][[sprintf("%s_ctx%d", mc, ng)]] <- v
            if (cfg$has_pron) {
                vp <- load_surp_safe(file.path(cfg$pron_dir,    m, sprintf("context_%d.txt", ng)), n)
                vb <- load_surp_safe(file.path(cfg$pron_bl_dir, m, sprintf("context_%d.txt", ng)), n)
                if (!is.null(vp)) preds_list[[corp]][[sprintf("%s_ctx%d_pron",          mc, ng)]] <- vp
                if (!is.null(vb)) preds_list[[corp]][[sprintf("%s_ctx%d_pron_baseline", mc, ng)]] <- vb
            }
        }
    }
    message(sprintf("  [%s] %d rows, %d cols", corp, nrow(preds_list[[corp]]), ncol(preds_list[[corp]])))
}

# Sanity check: correlation and mean surprisal
message("Computing sanity check metrics...")
sanity_rows <- list()
for (corp in names(corpus_cfg)) {
    preds  <- preds_list[[corp]]
    uni    <- preds[["unisurp"]]
    for (m in model_order) {
        mc     <- gsub("[-.]", "_", m)
        ctx_df <- contexts_per_model[[m]]
        for (i in seq_len(nrow(ctx_df))) {
            col <- sprintf("%s_ctx%d", mc, ctx_df$n_gram[i])
            if (!col %in% names(preds)) next
            vals  <- preds[[col]]
            n_obs <- sum(!is.na(vals))
            se    <- sd(vals, na.rm = TRUE) / sqrt(n_obs)
            sanity_rows[[length(sanity_rows) + 1]] <- data.frame(
                corpus = corp, model = m, context = ctx_df$context_size[i],
                cor_with_unisurp = cor(vals, uni, use = "complete.obs"),
                mean_surp  = mean(vals, na.rm = TRUE),
                lower_surp = mean(vals, na.rm = TRUE) - 1.96 * se,
                upper_surp = mean(vals, na.rm = TRUE) + 1.96 * se,
                stringsAsFactors = FALSE
            )
        }
    }
}
sanity_df <- bind_rows(sanity_rows)
dir.create("../result", showWarnings = FALSE, recursive = TRUE)
write.csv(sanity_df, "../result/sanity_check.csv", row.names = FALSE)
message("Saved: ../result/sanity_check.csv")

# Spillover and z-scoring
NO_SPILL <- c("story", "zone", "word", "bos", "eos", "is_punct", "position", "pos", "surp")

build_spillover <- function(df) {
    sp_cols <- setdiff(names(df), NO_SPILL)
    make_shift <- function(offset, sfx)
        df %>% mutate(zone = zone + offset) %>%
        select(story, zone, all_of(sp_cols)) %>%
        rename_with(~ paste0(., sfx), .cols = all_of(sp_cols))
    df %>%
        merge(make_shift(1L, "_so1"), by = c("story", "zone"), sort = FALSE) %>%
        merge(make_shift(2L, "_so2"), by = c("story", "zone"), sort = FALSE) %>%
        filter(is_punct == 0, bos == 0, eos == 0) %>%
        select(-is_punct, -bos, -eos) %>%
        drop_na()
}

scale_df <- function(df) {
    sc <- grep("^(zone|wlen|unisurp)|_ctx\\d+", names(df), value = TRUE, perl = TRUE)
    df[sc] <- lapply(df[sc], function(x) as.numeric(scale(x)))
    df
}

message("Building spillover terms...")
preds_list[["os"]] <- preds_list[["os"]] %>%
    mutate(story = paste(article_batch, article_id, difficulty_level, sep = "-")) %>%
    select(-article_batch, -article_id, -difficulty_level)

pred_proc <- lapply(preds_list, build_spillover)
for (corp in names(pred_proc))
    message(sprintf("  [%s] after spillover: %d rows", corp, nrow(pred_proc[[corp]])))

load_rt <- list(
    ns_spr = function()
        read.table("../data/naturalstories/naturalstories_RTS/processed_RTS.tsv", sep = "\t", quote = "", header = TRUE) %>%
        select(item, zone, RT) %>% rename(story = item) %>%
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

    osff = function()
        read_csv("../data/OneStop/rts.csv", col_types = cols_only(article_batch = "c", article_id = "c", difficulty_level = "c", zone = "d", IA_FIRST_FIXATION_DURATION = "d"), na = c("", "NA", ".")) %>%
        mutate(story = paste(article_batch, article_id, difficulty_level, sep = "-")) %>%
        drop_na(IA_FIRST_FIXATION_DURATION) %>%
        select(-article_batch, -article_id, -difficulty_level) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(IA_FIRST_FIXATION_DURATION, na.rm = TRUE), .groups = "drop"),

    osgd = function()
        read_csv("../data/OneStop/rts.csv", col_types = cols_only(article_batch = "c", article_id = "c", difficulty_level = "c", zone = "d", IA_FIRST_RUN_DWELL_TIME = "d"), na = c("", "NA", ".")) %>%
        mutate(story = paste(article_batch, article_id, difficulty_level, sep = "-")) %>%
        drop_na(IA_FIRST_RUN_DWELL_TIME) %>%
        select(-article_batch, -article_id, -difficulty_level) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(IA_FIRST_RUN_DWELL_TIME, na.rm = TRUE), .groups = "drop"),

    ostf = function()
        read_csv("../data/OneStop/rts.csv", col_types = cols_only(article_batch = "c", article_id = "c", difficulty_level = "c", zone = "d", IA_DWELL_TIME = "d"), na = c("", "NA", ".")) %>%
        mutate(story = paste(article_batch, article_id, difficulty_level, sep = "-")) %>%
        drop_na(IA_DWELL_TIME) %>%
        select(-article_batch, -article_id, -difficulty_level) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(IA_DWELL_TIME, na.rm = TRUE), .groups = "drop"),

    provoff = function()
        read_csv("../data/provo_corpus/Provo_Corpus-Eyetracking_Data.csv", col_types = cols_only(Text_ID = "d", Word_Number = "d", IA_FIRST_FIXATION_DURATION = "d")) %>%
        rename(story = Text_ID, zone = Word_Number) %>%
        drop_na(IA_FIRST_FIXATION_DURATION) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(IA_FIRST_FIXATION_DURATION, na.rm = TRUE), .groups = "drop"),

    provogd = function()
        read_csv("../data/provo_corpus/Provo_Corpus-Eyetracking_Data.csv", col_types = cols_only(Text_ID = "d", Word_Number = "d", IA_FIRST_RUN_DWELL_TIME = "d")) %>%
        rename(story = Text_ID, zone = Word_Number) %>%
        drop_na(IA_FIRST_RUN_DWELL_TIME) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(IA_FIRST_RUN_DWELL_TIME, na.rm = TRUE), .groups = "drop"),

    provotf = function()
        read_csv("../data/provo_corpus/Provo_Corpus-Eyetracking_Data.csv", col_types = cols_only(Text_ID = "d", Word_Number = "d", IA_DWELL_TIME = "d")) %>%
        rename(story = Text_ID, zone = Word_Number) %>%
        drop_na(IA_DWELL_TIME) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(IA_DWELL_TIME, na.rm = TRUE), .groups = "drop")
)

ds_config <- list(
    ns_spr  = list(corpus = "ns"),
    ns_maze = list(corpus = "ns"),
    brown   = list(corpus = "brown"),
    osff    = list(corpus = "os"),
    osgd    = list(corpus = "os"),
    ostf    = list(corpus = "os"),
    provoff = list(corpus = "provo"),
    provogd = list(corpus = "provo"),
    provotf = list(corpus = "provo")
)

df_list <- setNames(lapply(names(ds_config), function(ds) {
    cfg <- ds_config[[ds]]
    rt  <- load_rt[[ds]]()
    scale_df(merge(rt, pred_proc[[cfg$corpus]], by = c("story", "zone"), sort = FALSE) %>% ungroup() %>% drop_na())
}), names(ds_config))
for (ds in names(df_list))
    message(sprintf("[%s] after RT merge: %d rows", ds, nrow(df_list[[ds]])))

n_folds <- 10L
n_cores <- max(1L, parallel::detectCores() - 2L)
outcome <- "mean_RT"


# DLL for original texts — 9 datasets, 2 baselines (w/o and w/ unisurp)
message(sprintf("\n── Original DLL (%d datasets, %d-fold CV, %d cores) ──", length(ds_config), n_folds, n_cores))
all_results_orig <- list()

for (ds_name in names(ds_config)) {
    message(sprintf("\n=== Dataset: %s ===", ds_name))
    df   <- df_list[[ds_name]]
    corp <- ds_config[[ds_name]]$corpus

    set.seed(5963)
    folds    <- sample(rep(seq_len(n_folds), length.out = nrow(df)))
    idx_list <- lapply(seq_len(n_folds), function(k)
        list(tr = which(folds != k), te = which(folds == k)))

    cache_base <- function(f) lapply(seq_len(n_folds), function(k) {
        m <- lm(f, data = df[idx_list[[k]]$tr, ])
        list(pred  = predict(m, df[idx_list[[k]]$te, ]),
             sigma = sigma(m),
             y     = df[[outcome]][idx_list[[k]]$te])
    })
    base1_cache <- cache_base(as.formula(sprintf(
        "%s ~ zone + wlen + wlen_so1 + wlen_so2", outcome)))
    base2_cache <- cache_base(as.formula(sprintf(
        "%s ~ zone + wlen + wlen_so1 + wlen_so2 + unisurp + unisurp_so1 + unisurp_so2", outcome)))

    param_grid <- do.call(rbind, lapply(model_order, function(m) {
        mc     <- gsub("[-.]", "_", m)
        ctx_df <- contexts_per_model[[m]]
        valid  <- sapply(seq_len(nrow(ctx_df)), function(i) {
            ng <- ctx_df$n_gram[i]
            all(sprintf("%s_ctx%d%s", mc, ng, c("", "_so1", "_so2")) %in% names(df))
        })
        if (!any(valid)) return(NULL)
        data.frame(model = m, n_gram = ctx_df$n_gram[valid], context = ctx_df$context_size[valid],
                   stringsAsFactors = FALSE)
    }))
    if (corp == "provo")
        param_grid <- param_grid[param_grid$context <= PROVO_MAX_CTX, ]
    message(sprintf("  %d (model x context) combinations", nrow(param_grid)))

    cl   <- parallel::makeCluster(n_cores)
    doSNOW::registerDoSNOW(cl)
    pb   <- txtProgressBar(min = 0, max = nrow(param_grid), style = 3)
    opts <- list(progress = function(i) setTxtProgressBar(pb, i))

    res <- foreach::foreach(
        i             = seq_len(nrow(param_grid)),
        .combine      = rbind,
        .packages     = "stats",
        .export       = c("df", "idx_list", "base1_cache", "base2_cache", "n_folds", "outcome"),
        .options.snow = opts
    ) %dopar% {
        pg  <- param_grid[i, ]
        mc  <- gsub("[-.]", "_", pg$model)
        ng  <- as.integer(pg$n_gram)
        ctx <- as.integer(pg$context)

        smry <- function(vals) {
            n <- sum(!is.na(vals)); mn <- mean(vals, na.rm = TRUE)
            se <- if (n > 1L) sd(vals, na.rm = TRUE) / sqrt(n) else NA_real_
            c(mean = mn, lower = mn - 1.96 * se, upper = mn + 1.96 * se)
        }
        dll_vs_cache <- function(cache, tgt_f)
            sapply(seq_len(n_folds), function(k) {
                y   <- cache[[k]]$y
                m_t <- lm(tgt_f, data = df[idx_list[[k]]$tr, ])
                mean(dnorm(y, predict(m_t, df[idx_list[[k]]$te, ]), sigma(m_t), log = TRUE) -
                     dnorm(y, cache[[k]]$pred, cache[[k]]$sigma, log = TRUE), na.rm = TRUE)
            })

        make_f <- function(extra) as.formula(sprintf(
            "%s ~ zone + wlen + wlen_so1 + wlen_so2%s + %s_ctx%d + %s_ctx%d_so1 + %s_ctx%d_so2",
            outcome, extra, mc, ng, mc, ng, mc, ng))

        r1 <- smry(dll_vs_cache(base1_cache, make_f("")))
        r2 <- smry(dll_vs_cache(base2_cache, make_f("+ unisurp + unisurp_so1 + unisurp_so2")))

        data.frame(model = pg$model, context = ctx, n_folds = n_folds,
                   mean_dll_no_unisurp   = r1["mean"], lower_ci_no_unisurp   = r1["lower"],
                   upper_ci_no_unisurp   = r1["upper"],
                   mean_dll_with_unisurp = r2["mean"], lower_ci_with_unisurp = r2["lower"],
                   upper_ci_with_unisurp = r2["upper"],
                   stringsAsFactors = FALSE)
    }
    close(pb)
    parallel::stopCluster(cl)
    rownames(res) <- NULL
    all_results_orig[[ds_name]] <- res %>% mutate(dataset = ds_name, corpus = corp)
}

results_orig <- bind_rows(all_results_orig) %>% mutate(model = factor(model, levels = model_order))
write.csv(results_orig, "../result/lm_dll.csv", row.names = FALSE)
message("Saved: ../result/lm_dll.csv")


# DLL comparison — original vs pronominalized (all 9 datasets)
message(sprintf("\n── Pronominalized DLL (%d datasets) ──", length(ds_config)))
base_formula <- as.formula(paste(
    outcome, "~ zone + wlen + wlen_so1 + wlen_so2 + unisurp + unisurp_so1 + unisurp_so2"))

all_results_pron <- list()

for (ds_name in names(ds_config)) {
    message(sprintf("\n=== Dataset: %s ===", ds_name))
    df <- df_list[[ds_name]]

    set.seed(1)
    folds    <- sample(rep(seq_len(n_folds), length.out = nrow(df)))
    idx_list <- lapply(seq_len(n_folds), function(k)
        list(tr = which(folds != k), te = which(folds == k)))

    base_cache <- lapply(seq_len(n_folds), function(k) {
        m <- lm(base_formula, data = df[idx_list[[k]]$tr, ])
        list(pred  = predict(m, df[idx_list[[k]]$te, ]),
             sigma = sigma(m),
             y     = df[[outcome]][idx_list[[k]]$te])
    })

    param_grid <- do.call(rbind, lapply(model_order, function(m_name) {
        mc     <- gsub("[-.]", "_", m_name)
        ctx_df <- contexts_per_model[[m_name]]
        rows   <- lapply(seq_len(nrow(ctx_df)), function(i) {
            ng          <- ctx_df$n_gram[i]
            col_orig    <- sprintf("%s_ctx%d",              mc, ng)
            col_pron    <- sprintf("%s_ctx%d_pron",         mc, ng)
            col_pron_bl <- sprintf("%s_ctx%d_pron_baseline", mc, ng)
            has_orig    <- all(paste0(col_orig,    c("", "_so1", "_so2")) %in% names(df))
            has_pron    <- all(paste0(col_pron,    c("", "_so1", "_so2")) %in% names(df))
            has_pron_bl <- all(paste0(col_pron_bl, c("", "_so1", "_so2")) %in% names(df))
            if (!has_orig && !has_pron) return(NULL)
            data.frame(model = m_name, n_gram = ng, context = ctx_df$context_size[i],
                       has_orig = has_orig, has_pron = has_pron, has_pron_bl = has_pron_bl,
                       stringsAsFactors = FALSE)
        })
        do.call(rbind, Filter(Negate(is.null), rows))
    }))

    if (is.null(param_grid) || nrow(param_grid) == 0) {
        message("  No valid combinations, skipping."); next
    }
    if (ds_config[[ds_name]]$corpus == "provo")
        param_grid <- param_grid[param_grid$context <= PROVO_MAX_CTX, ]
    if (nrow(param_grid) == 0) { message("  No valid combinations after Provo cap, skipping."); next }
    message(sprintf("  %d (model x context) combinations", nrow(param_grid)))

    cl   <- parallel::makeCluster(n_cores)
    doSNOW::registerDoSNOW(cl)
    pb   <- txtProgressBar(min = 0, max = nrow(param_grid), style = 3)
    opts <- list(progress = function(i) setTxtProgressBar(pb, i))

    res <- foreach::foreach(
        i             = seq_len(nrow(param_grid)),
        .combine      = rbind,
        .packages     = "stats",
        .export       = c("df", "idx_list", "base_cache", "n_folds", "outcome"),
        .options.snow = opts
    ) %dopar% {
        pg          <- param_grid[i, ]
        mc          <- gsub("[-.]", "_", pg$model)
        ng          <- as.integer(pg$n_gram)
        ctx         <- as.integer(pg$context)
        has_orig    <- as.logical(pg$has_orig)
        has_pron    <- as.logical(pg$has_pron)
        has_pron_bl <- as.logical(pg$has_pron_bl)

        smry <- function(vals) {
            n <- sum(!is.na(vals)); mn <- mean(vals, na.rm = TRUE)
            se <- if (n > 1L) sd(vals, na.rm = TRUE) / sqrt(n) else NA_real_
            c(mean = mn, lower = mn - 1.96 * se, upper = mn + 1.96 * se)
        }
        dll_for_prefix <- function(col_prefix) {
            tgt_f <- as.formula(sprintf(
                "%s ~ zone + wlen + wlen_so1 + wlen_so2 + unisurp + unisurp_so1 + unisurp_so2 + %s + %s_so1 + %s_so2",
                outcome, col_prefix, col_prefix, col_prefix))
            sapply(seq_len(n_folds), function(k) {
                y   <- base_cache[[k]]$y
                m_t <- lm(tgt_f, data = df[idx_list[[k]]$tr, ])
                mean(dnorm(y, predict(m_t, df[idx_list[[k]]$te, ]), sigma(m_t), log = TRUE) -
                     dnorm(y, base_cache[[k]]$pred, base_cache[[k]]$sigma, log = TRUE), na.rm = TRUE)
            })
        }

        col_orig    <- sprintf("%s_ctx%d",               mc, ng)
        col_pron    <- sprintf("%s_ctx%d_pron",          mc, ng)
        col_pron_bl <- sprintf("%s_ctx%d_pron_baseline", mc, ng)

        dll_orig_folds    <- if (has_orig)    dll_for_prefix(col_orig)    else rep(NA_real_, n_folds)
        dll_pron_folds    <- if (has_pron)    dll_for_prefix(col_pron)    else rep(NA_real_, n_folds)
        dll_pron_bl_folds <- if (has_pron_bl) dll_for_prefix(col_pron_bl) else rep(NA_real_, n_folds)

        r_orig      <- smry(dll_orig_folds)
        r_pron      <- smry(dll_pron_folds)
        r_pron_bl   <- smry(dll_pron_bl_folds)
        r_delta     <- smry(dll_orig_folds - dll_pron_folds)
        r_delta_bl  <- smry(dll_orig_folds - dll_pron_bl_folds)
        r_pron_diff <- smry(dll_pron_bl_folds - dll_pron_folds)  # >0 when baseline DLL > linking DLL

        data.frame(
            model = pg$model, context = ctx,
            mean_dll_orig           = r_orig["mean"],      lower_orig           = r_orig["lower"],      upper_orig           = r_orig["upper"],
            mean_dll_pron           = r_pron["mean"],      lower_pron           = r_pron["lower"],      upper_pron           = r_pron["upper"],
            mean_dll_pron_baseline  = r_pron_bl["mean"],   lower_pron_baseline  = r_pron_bl["lower"],   upper_pron_baseline  = r_pron_bl["upper"],
            mean_dll_delta          = r_delta["mean"],     lower_delta          = r_delta["lower"],      upper_delta          = r_delta["upper"],
            mean_dll_delta_baseline = r_delta_bl["mean"],  lower_delta_baseline = r_delta_bl["lower"],   upper_delta_baseline = r_delta_bl["upper"],
            mean_dll_pron_diff      = r_pron_diff["mean"], lower_pron_diff      = r_pron_diff["lower"],  upper_pron_diff      = r_pron_diff["upper"],
            stringsAsFactors = FALSE
        )
    }
    close(pb)
    parallel::stopCluster(cl)
    rownames(res) <- NULL
    all_results_pron[[ds_name]] <- res %>% mutate(dataset = ds_name)
}

results_pron <- bind_rows(all_results_pron) %>% mutate(model = factor(model, levels = model_order))
write.csv(results_pron, "../result/lm_dll_pronominalized.csv", row.names = FALSE)
message("Saved: ../result/lm_dll_pronominalized.csv")
