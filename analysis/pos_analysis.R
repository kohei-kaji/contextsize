library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(brms)
library(ggdist)

set.seed(12345)

# ── Constants ─────────────────────────────────────────────────────────────────
N_FOLDS       <- 10L
CS_SHORT      <- 2L
CS_LONG       <- 1023L
POS_MIN_N     <- 100L
OUTCOME       <- "mean_RT"
SURP_BASE     <- "../outputs/inference_results"
MC            <- "gpt2"

STAGE <- c(1L, 2L)

CSV_SURP <- "../result/word_delta_surp.csv"
CSV_DLL  <- "../result/word_dll.csv"

POS_ORDER  <- c("NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV", "NUM", "DET", "PRON", "ADP", "CCONJ", "SCONJ", "PART")
dir_levels <- c("Longer is better", "Shorter is better")
clr        <- c("Longer is better" = "#F8766D", "Shorter is better" = "#00BFC4")

corpus_cfg <- list(
    ns    = list(preds_path = "../data/baselines_ns.csv",
                 surp_dir   = file.path(SURP_BASE, "ns"),
                 dep_path   = "../data/dep_ns.txt"),
    brown = list(preds_path = "../data/brown_spr/preds.csv",
                 surp_dir   = file.path(SURP_BASE, "brown"),
                 dep_path   = "../data/dep_brown.txt"),
    os    = list(preds_path = "../data/OneStop/preds.csv",
                 surp_dir   = file.path(SURP_BASE, "os"),
                 dep_path   = "../data/dep_os.txt"),
    provo = list(preds_path = "../data/provo_corpus/preds.csv",
                 surp_dir   = file.path(SURP_BASE, "provo"),
                 dep_path   = "../data/dep_provo.txt")
)

ds_config <- list(
    ns_spr  = list(corpus = "ns"),    ns_maze = list(corpus = "ns"),
    brown   = list(corpus = "brown"),
    osff    = list(corpus = "os"),    osgd    = list(corpus = "os"),    ostf    = list(corpus = "os"),
    provoff = list(corpus = "provo"), provogd = list(corpus = "provo"), provotf = list(corpus = "provo")
)

load_rt <- list(
    ns_spr = function()
        read.table("../data/naturalstories/naturalstories_RTS/processed_RTS.tsv",
                   sep = "\t", quote = "", header = TRUE) %>%
        select(item, zone, RT) %>% rename(story = item) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(RT, na.rm = TRUE), .groups = "drop"),

    ns_maze = function()
        read_rds("../data/maze/maze.rds") %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(maze_RT, na.rm = TRUE), .groups = "drop"),

    brown = function()
        read_csv("../data/brown_spr/brown_spr.csv",
                 col_types = cols_only(text_id = "d", text_pos = "d", time = "d")) %>%
        rename(story = text_id, zone = text_pos) %>%
        mutate(story = story + 1L, zone = zone + 1L) %>%
        filter(time > 100, time <= 3000) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(time, na.rm = TRUE), .groups = "drop"),

    osff = function()
        read_csv("../data/OneStop/rts.csv", col_types = cols_only(article_batch="c",article_id="c",difficulty_level="c", zone="d",IA_FIRST_FIXATION_DURATION="d"), na = c("","NA",".")) %>%
        mutate(story = paste(article_batch, article_id, difficulty_level, sep="-")) %>%
        drop_na(IA_FIRST_FIXATION_DURATION) %>%
        select(-article_batch,-article_id,-difficulty_level) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(IA_FIRST_FIXATION_DURATION, na.rm=TRUE), .groups="drop"),

    osgd = function()
        read_csv("../data/OneStop/rts.csv", col_types = cols_only(article_batch="c",article_id="c",difficulty_level="c", zone="d",IA_FIRST_RUN_DWELL_TIME="d"), na = c("","NA",".")) %>%
        mutate(story = paste(article_batch, article_id, difficulty_level, sep="-")) %>%
        drop_na(IA_FIRST_RUN_DWELL_TIME) %>%
        select(-article_batch,-article_id,-difficulty_level) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(IA_FIRST_RUN_DWELL_TIME, na.rm=TRUE), .groups="drop"),

    ostf = function()
        read_csv("../data/OneStop/rts.csv", col_types = cols_only(article_batch="c",article_id="c",difficulty_level="c", zone="d",IA_DWELL_TIME="d"), na = c("","NA",".")) %>%
        mutate(story = paste(article_batch, article_id, difficulty_level, sep="-")) %>%
        drop_na(IA_DWELL_TIME) %>%
        select(-article_batch,-article_id,-difficulty_level) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(IA_DWELL_TIME, na.rm=TRUE), .groups="drop"),

    provoff = function()
        read_csv("../data/provo_corpus/Provo_Corpus-Eyetracking_Data.csv", col_types = cols_only(Text_ID="d",Word_Number="d",IA_FIRST_FIXATION_DURATION="d")) %>%
        rename(story=Text_ID, zone=Word_Number) %>%
        drop_na(IA_FIRST_FIXATION_DURATION) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(IA_FIRST_FIXATION_DURATION, na.rm=TRUE), .groups="drop"),

    provogd = function()
        read_csv("../data/provo_corpus/Provo_Corpus-Eyetracking_Data.csv", col_types = cols_only(Text_ID="d",Word_Number="d",IA_FIRST_RUN_DWELL_TIME="d")) %>%
        rename(story=Text_ID, zone=Word_Number) %>%
        drop_na(IA_FIRST_RUN_DWELL_TIME) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(IA_FIRST_RUN_DWELL_TIME, na.rm=TRUE), .groups="drop"),

    provotf = function()
        read_csv("../data/provo_corpus/Provo_Corpus-Eyetracking_Data.csv", col_types = cols_only(Text_ID="d",Word_Number="d",IA_DWELL_TIME="d")) %>%
        rename(story=Text_ID, zone=Word_Number) %>%
        drop_na(IA_DWELL_TIME) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(IA_DWELL_TIME, na.rm=TRUE), .groups="drop")
)

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

scale_numeric_cols <- function(df) {
    sc <- grep("^(zone|wlen|unisurp)|_ctx\\d+", names(df), value = TRUE, perl = TRUE)
    df[sc] <- lapply(df[sc], function(x) as.numeric(scale(x)))
    df
}

normalize_str <- function(x) tolower(gsub("[^a-z0-9]", "", x, perl = TRUE))

align_global_multi <- function(preds_words, dep_words, dep_pos) {
    p_norm <- normalize_str(preds_words)
    d_norm <- normalize_str(dep_words)
    p_cum  <- c(0L, cumsum(nchar(p_norm)))
    d_cum  <- c(0L, cumsum(nchar(d_norm)))
    n_d    <- length(dep_words)
    lapply(seq_along(preds_words), function(i) {
        p_start <- p_cum[i]
        p_end   <- p_cum[i + 1L]
        overlap <- which(d_cum[-length(d_cum)] < p_end & d_cum[-1L] > p_start)
        if (length(overlap) == 0L)
            overlap <- pmin(pmax(findInterval(p_start, d_cum), 1L), n_d)
        unique(dep_pos[overlap])
    })
}

parse_dep_file <- function(filepath) {
    lines     <- readLines(filepath, warn = FALSE)
    tok_lines <- lines[grepl("^[0-9]+\t", lines)]
    tok_lines <- tok_lines[!grepl("^[0-9]+-", tok_lines) & !grepl("^[0-9]+\\.", tok_lines)]
    parts     <- strsplit(tok_lines, "\t")
    data.frame(ud_word = sapply(parts, `[`, 2L),
               ud_pos  = sapply(parts, `[`, 4L),
               stringsAsFactors = FALSE)
}

compute_dll_two_W <- function(df, col_short, col_long) {
    make_f <- function(surp_col) as.formula(sprintf(
        "%s ~ zone + wlen + wlen_so1 + wlen_so2 + unisurp + unisurp_so1 + unisurp_so2 + %s + %s_so1 + %s_so2",
        OUTCOME, surp_col, surp_col, surp_col))
    f_short  <- make_f(col_short)
    f_long   <- make_f(col_long)
    folds    <- sample(rep(seq_len(N_FOLDS), length.out = nrow(df)))
    ll_short <- ll_long <- numeric(nrow(df))
    for (k in seq_len(N_FOLDS)) {
        tr <- which(folds != k); te <- which(folds == k)
        m_short <- lm(f_short, data = df[tr, ])
        m_long  <- lm(f_long,  data = df[tr, ])
        y <- df[[OUTCOME]][te]
        ll_short[te] <- dnorm(y, predict(m_short, df[te, ]), sigma(m_short), log = TRUE)
        ll_long[te]  <- dnorm(y, predict(m_long,  df[te, ]), sigma(m_long),  log = TRUE)
    }
    data.frame(dll_diff = ll_long - ll_short)
}

extract_brms_draws <- function(fit, prefix = "b_ud_pos") {
    vars <- grep(paste0("^", prefix), variables(fit), value = TRUE)
    as_draws_df(fit, variable = vars) %>%
        as_tibble() %>%
        select(all_of(vars)) %>%
        pivot_longer(all_of(vars), names_to = "param", values_to = "draw") %>%
        mutate(ud_pos = sub(paste0("^", prefix), "", param)) %>%
        select(ud_pos, draw)
}

extract_brms_coefs <- function(fit, prefix = "b_ud_pos") {
    extract_brms_draws(fit, prefix) %>%
        group_by(ud_pos) %>%
        summarise(
            estimate = median(draw),
            lower_ci = quantile(draw, 0.025),
            upper_ci = quantile(draw, 0.975),
            prob_pos = mean(draw > 0),
            .groups  = "drop"
        )
}


add_plot_cols <- function(df, pos_rev = TRUE) {
    pos_lev <- POS_ORDER[POS_ORDER %in% unique(df$ud_pos)]
    if (pos_rev) pos_lev <- rev(pos_lev)
    df %>% mutate(
        sig        = (prob_pos > 0.975 | prob_pos < 0.025),
        direction  = factor(ifelse(estimate > 0, "Longer is better", "Shorter is better"), levels = dir_levels),
        fill_color = ifelse(sig, as.character(direction), "ns"),
        ud_pos_ord = factor(ud_pos, levels = pos_lev)
    )
}

make_pos_plot_bayes <- function(draws_df, coefs_df, x_label) {
    coefs_aug    <- coefs_df %>% filter(ud_pos %in% POS_ORDER) %>% add_plot_cols()
    pos_lev      <- levels(coefs_aug$ud_pos_ord)
    present_dirs <- intersect(dir_levels, unique(as.character(coefs_aug$direction)))

    plot_draws <- draws_df %>%
        filter(ud_pos %in% POS_ORDER) %>%
        left_join(coefs_aug %>% select(ud_pos, fill_color, ud_pos_ord), by = "ud_pos") %>%
        group_by(ud_pos) %>%
        filter(draw >= quantile(draw, 0.005) & draw <= quantile(draw, 0.995)) %>%
        ungroup()

    ggplot(plot_draws, aes(y = ud_pos_ord, x = draw, fill = fill_color)) +
        stat_halfeye(
            .width          = c(0.80, 0.95),
            point_interval  = median_qi,
            normalize       = "groups",
            color           = "gray30",
            point_color     = "gray20",
            interval_color  = "gray20"
        ) +
        geom_point(data        = coefs_aug,
                   aes(x = estimate, y = ud_pos_ord, color = direction),
                   shape       = 21, size = 2.5, stroke = 0.8,
                   fill        = NA, inherit.aes = FALSE) +
        geom_vline(xintercept = 0, linetype = "dashed",
                   color = "gray50", linewidth = 0.5) +
        scale_fill_manual(values = c(clr, ns = "gray82"), guide = "none") +
        scale_color_manual(values = clr, name = NULL, breaks = present_dirs) +
        labs(y = NULL, x = x_label, color = NULL) +
        theme_bw() +
        theme(strip.text      = element_text(face = "bold", size = 12),
              axis.text.y     = element_text(size = 12),
              axis.text.x     = element_text(size = 9),
              axis.title.x    = element_text(size = 15, margin = margin(t = 10)),
              legend.position = "none")
}

save_plot <- function(p, stem, width, height) {
    dir.create("../result/png", showWarnings = FALSE, recursive = TRUE)
    dir.create("../result/pdf", showWarnings = FALSE, recursive = TRUE)
    ggsave(sprintf("../result/png/%s.png", stem), p, width = width, height = height, dpi = 300)
    ggsave(sprintf("../result/pdf/%s.pdf", stem), p, width = width, height = height)
    message(sprintf("Saved: ../result/png/%s.png + .pdf", stem))
}

col_short <- sprintf("%s_ctx%d", MC, CS_SHORT + 1L)
col_long  <- sprintf("%s_ctx%d", MC, CS_LONG  + 1L)

dir.create("../result", showWarnings = FALSE, recursive = TRUE)


if (1L %in% STAGE) {
    message("\n══ Stage 1: Computing word-by-word data ══")

    message("Loading preds and surp files ...")
    preds_cache <- list()
    for (corp in names(corpus_cfg)) {
        cfg   <- corpus_cfg[[corp]]
        preds <- read_csv(cfg$preds_path, show_col_types = FALSE)
        if (corp == "os")
            preds <- preds %>%
                mutate(story = paste(article_batch, article_id, difficulty_level, sep="-")) %>%
                select(-article_batch, -article_id, -difficulty_level)
        n <- nrow(preds)
        for (ng in c(CS_SHORT + 1L, CS_LONG + 1L)) {
            col   <- sprintf("%s_ctx%d", MC, ng)
            fpath <- file.path(cfg$surp_dir, MC, sprintf("context_%d.txt", ng))
            vals  <- if (file.exists(fpath)) {
                v <- scan(fpath, what = numeric(), quiet = TRUE)
                if (length(v) == n) v
                else { warning(sprintf("[%s] row mismatch ctx%d", corp, ng)); rep(NA_real_, n) }
            } else { warning(sprintf("[%s] missing ctx%d", corp, ng)); rep(NA_real_, n) }
            preds[[col]] <- vals
        }
        preds_cache[[corp]] <- preds
        message(sprintf("  [%s] %d rows, %d cols", corp, n, ncol(preds)))
    }

    build_pos_map_simple <- function(corp) {
        preds_raw <- preds_cache[[corp]]
        cfg       <- corpus_cfg[[corp]]
        preds_ord <- if (corp == "os") {
            story_order <- preds_raw %>% distinct(story) %>% mutate(story_rank = row_number())
            preds_raw %>% left_join(story_order, by = "story") %>% arrange(story_rank, zone)
        } else {
            preds_raw %>% arrange(story, zone)
        }
        dep_tokens <- parse_dep_file(cfg$dep_path)
        pos_lists  <- align_global_multi(preds_ord$word, dep_tokens$ud_word, dep_tokens$ud_pos)
        data.frame(
            story  = as.character(preds_ord$story),
            zone   = preds_ord$zone,
            ud_pos = sapply(pos_lists, `[`, 1L),
            stringsAsFactors = FALSE
        )
    }
    pos_maps_simple <- setNames(lapply(names(corpus_cfg), build_pos_map_simple), names(corpus_cfg))

    message("\nDelta surprisal (word-by-word)")
    surp_rows <- bind_rows(lapply(names(corpus_cfg), function(corp) {
        preds_cache[[corp]] %>%
            filter(is_punct == 0, bos == 0, eos == 0) %>%
            select(story, zone, all_of(c(col_short, col_long))) %>%
            mutate(story = as.character(story)) %>%
            inner_join(pos_maps_simple[[corp]], by = c("story", "zone")) %>%
            filter(ud_pos %in% POS_ORDER,
                   !is.na(.data[[col_short]]), !is.na(.data[[col_long]])) %>%
            mutate(
                surp_short  = .data[[col_short]],
                surp_long   = .data[[col_long]],
                delta_surp  = .data[[col_short]] - .data[[col_long]],
                corpus      = corp
            ) %>%
            select(story, zone, corpus, ud_pos, surp_short, surp_long, delta_surp)
    }))
    write_csv(surp_rows, CSV_SURP)
    message(sprintf("Saved: %s (%d rows)", CSV_SURP, nrow(surp_rows)))

    message("\nDLL (word-by-word)")
    dll_rows_list <- list()
    for (ds_name in names(ds_config)) {
        corp <- ds_config[[ds_name]]$corpus
        message(sprintf("  %s ...", ds_name))
        if (!col_short %in% names(preds_cache[[corp]]) ||
            !col_long  %in% names(preds_cache[[corp]])) { message("  missing col — skip"); next }

        rt        <- load_rt[[ds_name]]()
        proc      <- build_spillover(preds_cache[[corp]])
        merged_df <- merge(rt, proc, by = c("story", "zone"), sort = FALSE) %>%
            ungroup() %>% drop_na()

        dll_df       <- compute_dll_two_W(scale_numeric_cols(merged_df), col_short, col_long)
        dll_df$story <- merged_df$story
        dll_df$zone  <- merged_df$zone

        dll_rows_list[[ds_name]] <- dll_df %>%
            mutate(story = as.character(story)) %>%
            inner_join(pos_maps_simple[[corp]], by = c("story", "zone")) %>%
            filter(ud_pos %in% POS_ORDER) %>%
            mutate(dataset = ds_name) %>%
            select(story, zone, dataset, ud_pos, dll_diff)
    }
    dll_rows <- bind_rows(dll_rows_list)
    write_csv(dll_rows, CSV_DLL)
    message(sprintf("Saved: %s (%d rows)", CSV_DLL, nrow(dll_rows)))

    message("\n══ Stage 1 complete ══")
}

if (2L %in% STAGE) {
    message("\n══ Stage 2: Regression analysis ══")

    total_cores <- parallel::detectCores()
    if (is.na(total_cores)) total_cores <- 4
    allowed_cores <- max(1L, total_cores - 2L)
    threads_per_chain <- max(1L, floor(allowed_cores / 4))
    message(sprintf("Total cores: %d | Allowed: %d | Threads per chain: %d (Total used: %d)", total_cores, allowed_cores, threads_per_chain, 4 * threads_per_chain))

    # ── Plot A: Delta surprisal by POS ────────────────────────────────────────
    message("\n── Plot A: Delta surprisal by POS (brms) ──")
    surp_pooled <- read_csv(CSV_SURP, show_col_types = FALSE) %>%
        group_by(ud_pos) %>% filter(n() >= POS_MIN_N) %>% ungroup() %>%
        mutate(ud_pos = factor(ud_pos))
    message(sprintf("  Read %s: %d rows", CSV_SURP, nrow(surp_pooled)))

    message("  Fitting brm: delta_surp ~ 0 + ud_pos + (0 + ud_pos | corpus) ...")
    fit_surp_single <- brm(
        delta_surp ~ 0 + ud_pos + (0 + ud_pos | corpus),
        data    = surp_pooled,
        family  = gaussian(),
        prior   = c(prior(normal(0, 2), class = b),
                    prior(exponential(1), class = sd),
                    prior(exponential(1), class = sigma)),
        chains  = 4,
        cores   = 4,
        threads = threading(threads_per_chain),
        normalize = FALSE,
        stan_model_args = list(stanc_options = list("O1")),
        iter    = 2000, warmup = 1000,
        seed    = 12345,
        silent  = 2,
        refresh = 0,
        backend = "cmdstanr",
        file    = "../result/fit_surp.rds"
    )
    draws_surp_single <- extract_brms_draws(fit_surp_single)
    coefs_surp_single <- extract_brms_coefs(fit_surp_single) %>%
        filter(ud_pos %in% POS_ORDER)

    write_csv(coefs_surp_single, "../result/pos_delta_surp_coefs.csv")
    message("Saved: ../result/pos_delta_surp_coefs.csv")

    save_plot(
        make_pos_plot_bayes(
            draws_surp_single, coefs_surp_single,
            "Reduction in Surprisal by Extending Context"
        ),
        "pos_delta_surp", width = 6, height = 4
    )

    # ── Plot B: Delta DLL by POS ──────────────────────────────────────────────
    message("\n── Plot B: Delta DLL by POS (brms) ──")
    dll_pooled <- read_csv(CSV_DLL, show_col_types = FALSE) %>%
        group_by(ud_pos) %>% filter(n() >= POS_MIN_N) %>% ungroup() %>%
        mutate(ud_pos = factor(ud_pos))
    message(sprintf("  Read %s: %d rows", CSV_DLL, nrow(dll_pooled)))

    message("  Fitting brm: dll_diff ~ 0 + ud_pos + (0 + ud_pos | dataset) ...")
    fit_dll_single <- brm(
        dll_diff ~ 0 + ud_pos + (0 + ud_pos | dataset),
        data    = dll_pooled,
        family  = gaussian(),
        prior   = c(prior(normal(0, 0.1), class = b),
                    prior(exponential(1), class = sd),
                    prior(exponential(1), class = sigma)),
        chains  = 4,
        cores   = 4,
        threads = threading(threads_per_chain),
        normalize = FALSE,
        stan_model_args = list(stanc_options = list("O1")),
        iter    = 2000, warmup = 1000,
        seed    = 12345,
        silent  = 2,
        refresh = 0,
        backend = "cmdstanr",
        file    = "../result/fit_dll.rds"
    )

    draws_dll_single <- extract_brms_draws(fit_dll_single)
    coefs_dll_single <- extract_brms_coefs(fit_dll_single) %>%
        filter(ud_pos %in% POS_ORDER)

    write_csv(coefs_dll_single, "../result/pos_dll_coefs.csv")
    message("Saved: ../result/pos_dll_coefs.csv")

    save_plot(
        make_pos_plot_bayes(
            draws_dll_single, coefs_dll_single,
            "Improvement in Predictive Power by Extending Context"
        ),
        "pos_dll", width = 6, height = 4
    )

    message("\n══ Stage 2 complete ══")
}
