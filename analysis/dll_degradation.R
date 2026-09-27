library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

# follow-up analysis of DLL degradation
results_pron <- read_csv("../result/lm_dll_pronominalized.csv", show_col_types = FALSE)

CANONICAL_ORDER <- c("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
models_in_data  <- unique(results_pron$model)
model_order     <- c(CANONICAL_ORDER[CANONICAL_ORDER %in% models_in_data], sort(setdiff(models_in_data, CANONICAL_ORDER)))
MODEL_LABELS    <- setNames(sub("^gpt2$", "gpt2-small", model_order), model_order)

DS_LABELS <- c(
    brown   = "Brown SPR",
    ns_spr  = "Natural Stories SPR",
    ns_maze = "Natural Stories A-Maze",
    osff    = "OneStop FF",
    osgd    = "OneStop GD",
    ostf    = "OneStop TF",
    provoff = "Provo FF",
    provogd = "Provo GD",
    provotf = "Provo TF"
)

results_pron <- results_pron %>%
    mutate(model = factor(model, levels = model_order))

dll_deg <- results_pron %>%
    filter(mean_dll_orig > 0) %>%
    mutate(
        pct_worse_linking   = mean_dll_delta          / mean_dll_orig * 100,
        pct_worse_singleton = mean_dll_delta_baseline / mean_dll_orig * 100
    )

out_per_row <- dll_deg %>%
    select(dataset, model, context,
           mean_dll_orig,
           mean_dll_pron, mean_dll_pron_baseline,
           pct_worse_linking, pct_worse_singleton)

dir.create("../result", showWarnings = FALSE, recursive = TRUE)
write_csv(out_per_row, "../result/dll_degradation.csv")
message("Saved: ../result/dll_degradation.csv")

# Summary: average across datasets, per model × context
summary_deg <- dll_deg %>%
    group_by(model, context) %>%
    summarise(
        n_datasets              = n(),
        mean_dll_orig           = mean(mean_dll_orig,        na.rm = TRUE),
        mean_pct_worse_linking  = mean(pct_worse_linking,    na.rm = TRUE),
        mean_pct_worse_singleton = mean(pct_worse_singleton, na.rm = TRUE),
        sd_pct_worse_linking    = sd(pct_worse_linking,      na.rm = TRUE),
        sd_pct_worse_singleton  = sd(pct_worse_singleton,    na.rm = TRUE),
        .groups = "drop"
    )

write_csv(summary_deg, "../result/dll_degradation_summary.csv")
message("Saved: ../result/dll_degradation_summary.csv")

cat("\n── Degradation summary (averaged across datasets) ──\n")
print(summary_deg %>% arrange(model, context), n = 100)

cat("\n── Context 1023 (Provo: 100): per-dataset breakdown (% worse) ──\n")
PROVO_DS_VEC <- c("provoff", "provogd", "provotf")
ctx1023 <- dll_deg %>%
    filter((!dataset %in% PROVO_DS_VEC & context == 1023L) |
           (dataset %in% PROVO_DS_VEC & context == 100L)) %>%
    mutate(dataset_label = DS_LABELS[dataset]) %>%
    select(model, dataset_label,
           mean_dll_orig,
           pct_worse_linking, pct_worse_singleton) %>%
    arrange(model, dataset_label)
print(ctx1023, n = 200)

# Plot of DLL degradation vs context size
MODEL_COLORS <- setNames(c("#FF4B00", "#005AFF", "#03AF7A", "#4DC4FF"), CANONICAL_ORDER)
model_colors <- MODEL_COLORS[model_order]

plot_df <- dll_deg %>%
    select(dataset, model, context, pct_worse_linking, pct_worse_singleton) %>%
    pivot_longer(c(pct_worse_linking, pct_worse_singleton), names_to  = "destruction", values_to = "pct_worse") %>%
    mutate(
        destruction   = recode(destruction,
                               pct_worse_linking   = "CorefDestroyed",
                               pct_worse_singleton = "SingletonDestroyed"),
        dataset_label = DS_LABELS[dataset]
    )

p_deg <- ggplot(plot_df, aes(x = context, y = pct_worse, color = model, group = interaction(model, dataset))) +
    geom_line(alpha = 0.5, linewidth = 0.5) +
    geom_point(size = 1.2, alpha = 0.7) +
    facet_wrap(~ destruction, ncol = 2) +
    scale_x_log10(
        breaks = c(0, 1, 2, 4, 7, 15, 31, 63, 127, 255, 511, 1023),
        labels = c(0, 1, 2, 4, 7, 15, 31, 63, 127, 255, 511, 1023)
    ) +
    scale_color_manual(values = model_colors, labels = MODEL_LABELS, name = "Model") +
    geom_hline(yintercept = 0,   linetype = "dashed", color = "gray50") +
    geom_hline(yintercept = 100, linetype = "dotted",  color = "gray70") +
    labs(x     = "Context size (tokens)",
         y     = "% of DLL benefit lost by Disrupted",
         title = "DLL degradation by context destruction") +
    theme_bw() +
    theme(strip.text      = element_text(face = "bold", size = 12),
          axis.title.x    = element_text(size = 14, margin = margin(t = 10)),
          axis.title.y    = element_text(size = 14, margin = margin(r = 10)),
          legend.text     = element_text(size = 12),
          legend.position = "bottom")

dir.create("../result/png", showWarnings = FALSE, recursive = TRUE)
dir.create("../result/pdf", showWarnings = FALSE, recursive = TRUE)
ggsave("../result/png/dll_degradation_pct.png", p_deg, width = 10, height = 5, dpi = 300)
ggsave("../result/pdf/dll_degradation_pct.pdf", p_deg, width = 10, height = 5)
message("Saved: ../result/png/dll_degradation_pct.png + .pdf")

# ── Box plot: % degradation at context 1023 (Provo: 100), gpt2 only ──────────
box_df <- dll_deg %>%
    filter(model == "gpt2",
           (dataset %in% PROVO_DS_VEC & context == 100L) |
           (!dataset %in% PROVO_DS_VEC & context == 1023L)) %>%
    select(dataset, pct_worse_linking, pct_worse_singleton) %>%
    pivot_longer(c(pct_worse_linking, pct_worse_singleton),
                 names_to  = "destruction",
                 values_to = "pct_worse") %>%
    mutate(destruction = recode(destruction,
                                pct_worse_linking   = "CorefDisrupted",
                                pct_worse_singleton = "SingletonDisrupted"))

p_box <- ggplot(box_df,
                aes(x = destruction, y = pct_worse, fill = destruction)) +
    geom_boxplot(width = 0.5, outlier.shape = NA, alpha = 1) +
    geom_jitter(width = 0.1, size = 2, alpha = 0.8, color = "gray30") +
    geom_hline(yintercept = 0,   linetype = "dashed", color = "gray50") +
    geom_hline(yintercept = 100, linetype = "dotted", color = "gray70") +
    scale_fill_manual(values = c(CorefDisrupted   = "#D55E00",
                                 SingletonDisrupted = "#005A9C"),
                      guide = "none") +
    labs(x     = NULL,
         y     = "% of Decrease in Log Likelihood") +
    theme_bw() +
    theme(axis.title.y  = element_text(size = 12, margin = margin(r = 10)),
          axis.text.x   = element_text(size = 10))

ggsave("../result/png/dll_degradation_maxctx.png", p_box, width = 4, height = 3, dpi = 300)
ggsave("../result/pdf/dll_degradation_maxctx.pdf", p_box, width = 4, height = 3)
message("Saved: ../result/png/dll_degradation_maxctx.png + .pdf")


# Word-level DLL t-tests: gpt2-small
#   ctx = 1023 for ns/brown/os;  ctx = 100 for provo
PROVO_DS      <- c("provoff", "provogd", "provotf")
PROVO_MAX_CTX <- 100L
NO_SPILL      <- c("story", "zone", "word", "bos", "eos", "is_punct", "position", "pos", "surp")
outcome  <- "mean_RT"
n_folds  <- 10L
mc       <- gsub("[-.]", "_", "gpt2")
surp_base <- "../data/surp"

spill <- function(df) {
    cols  <- setdiff(names(df), NO_SPILL)
    shift <- function(n, sfx)
        df %>% mutate(zone = zone + n) %>%
               select(story, zone, all_of(cols)) %>%
               rename_with(~paste0(., sfx), .cols = all_of(cols))
    df %>%
        merge(shift(1L, "_so1"), by = c("story", "zone"), sort = FALSE) %>%
        merge(shift(2L, "_so2"), by = c("story", "zone"), sort = FALSE) %>%
        filter(is_punct == 0, bos == 0, eos == 0) %>%
        select(-is_punct, -bos, -eos) %>%
        drop_na()
}

scl <- function(df) {
    cols <- grep("^(zone|wlen|unisurp)|_ctx\\d+", names(df), value = TRUE, perl = TRUE)
    df[cols] <- lapply(df[cols], function(x) as.numeric(scale(x)))
    df
}

preds_paths <- list(
    ns    = "../data/baselines_ns.csv",
    brown = "../data/brown_spr/preds.csv",
    os    = "../data/OneStop/preds.csv",
    provo = "../data/provo_corpus/preds.csv"
)

surp_dirs <- list(
    ns    = list(orig = "ns",    pron = "ns_pronominalized",      bl = "ns_baseline_pronominalized"),
    brown = list(orig = "brown", pron = "brown_pronominalized",   bl = "brown_baseline_pronominalized"),
    os    = list(orig = "os",    pron = "onestop_pronominalized", bl = "onestop_baseline_pronominalized"),
    provo = list(orig = "provo", pron = "provo_pronominalized",   bl = "provo_baseline_pronominalized")
)

load_rt_data <- list(
    ns_spr  = function()
        read.table("../data/naturalstories/naturalstories_RTS/processed_RTS.tsv",
                   sep = "\t", quote = "", header = TRUE) %>%
        rename(story = item) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(RT, na.rm = TRUE), .groups = "drop"),

    ns_maze = function()
        read_rds("../data/maze/maze.rds") %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(maze_RT, na.rm = TRUE), .groups = "drop"),

    brown   = function()
        read_csv("../data/brown_spr/brown_spr.csv",
                 col_types = cols_only(text_id = "d", text_pos = "d", time = "d")) %>%
        rename(story = text_id, zone = text_pos) %>%
        mutate(story = story + 1, zone = zone + 1) %>%
        filter(time > 100, time <= 3000) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(time, na.rm = TRUE), .groups = "drop"),

    osff    = function()
        read_csv("../data/OneStop/rts.csv", col_types = cols_only(article_batch = "c", article_id = "c", difficulty_level = "c", zone = "d", IA_FIRST_FIXATION_DURATION = "d"), na = c("", "NA", ".")) %>%
        drop_na(IA_FIRST_FIXATION_DURATION) %>%
        mutate(story = paste(article_batch, article_id, difficulty_level, sep = "-")) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(IA_FIRST_FIXATION_DURATION, na.rm = TRUE), .groups = "drop"),

    osgd    = function()
        read_csv("../data/OneStop/rts.csv",
                 col_types = cols_only(article_batch = "c", article_id = "c", difficulty_level = "c", zone = "d", IA_FIRST_RUN_DWELL_TIME = "d"),
                 na = c("", "NA", ".")) %>%
        drop_na(IA_FIRST_RUN_DWELL_TIME) %>%
        mutate(story = paste(article_batch, article_id, difficulty_level, sep = "-")) %>%
        group_by(story, zone) %>%
        summarise(mean_RT = mean(IA_FIRST_RUN_DWELL_TIME, na.rm = TRUE), .groups = "drop"),

    ostf    = function()
        read_csv("../data/OneStop/rts.csv",
                 col_types = cols_only(article_batch = "c", article_id = "c", difficulty_level = "c", zone = "d", IA_DWELL_TIME = "d"),
                 na = c("", "NA", ".")) %>%
        drop_na(IA_DWELL_TIME) %>%
        mutate(story = paste(article_batch, article_id, difficulty_level, sep = "-")) %>%
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

ds_to_corpus <- c(
    ns_spr = "ns", ns_maze = "ns", brown = "brown",
    osff   = "os", osgd    = "os", ostf  = "os",
    provoff = "provo", provogd = "provo", provotf = "provo"
)

# Pre-process preds for each corpus (attach surp columns + build spillover)
message("Pre-processing preds for word-level t-tests...")
preds_spill <- lapply(names(surp_dirs), function(corp) {
    ctx  <- if (corp == "provo") PROVO_MAX_CTX else 1023L
    ng   <- ctx + 1L
    dirs <- surp_dirs[[corp]]

    p <- read_csv(preds_paths[[corp]], show_col_types = FALSE)
    if (corp == "os")
        p <- p %>%
            mutate(story = paste(article_batch, article_id, difficulty_level, sep = "-")) %>%
            select(-article_batch, -article_id, -difficulty_level)

    col_o <- sprintf("%s_ctx%d",               mc, ng)
    col_p <- sprintf("%s_ctx%d_pron",          mc, ng)
    col_b <- sprintf("%s_ctx%d_pron_baseline", mc, ng)
    p[[col_o]] <- scan(file.path(surp_base, dirs$orig, "gpt2", sprintf("context_%d.txt", ng)), quiet = TRUE)
    p[[col_p]] <- scan(file.path(surp_base, dirs$pron, "gpt2", sprintf("context_%d.txt", ng)), quiet = TRUE)
    p[[col_b]] <- scan(file.path(surp_base, dirs$bl,   "gpt2", sprintf("context_%d.txt", ng)), quiet = TRUE)

    spill(p)
})
names(preds_spill) <- names(surp_dirs)


surp_f <- function(col) as.formula(sprintf(paste(
    outcome,
    "~ zone + wlen + wlen_so1 + wlen_so2 + unisurp + unisurp_so1 + unisurp_so2 + %s + %s_so1 + %s_so2"),
    col, col, col))

cat("\n── Word-level DLL t-tests (gpt2-small) ──\n")
cat("   Test 1: mean(dll_pron - dll_orig) < 0?\n")
cat("   Test 2: mean(dll_pron - dll_pron_bl) < 0?\n")
cat(sprintf("  %-10s  %5s  %9s  %10s    %9s  %10s  (n words)\n",
            "dataset", "ctx", "t1_t", "t1_p", "t2_t", "t2_p"))

for (ds in names(ds_to_corpus)) {
    corp <- ds_to_corpus[ds]
    ctx  <- if (ds %in% PROVO_DS) PROVO_MAX_CTX else 1023L
    ng   <- ctx + 1L
    col_o <- sprintf("%s_ctx%d",               mc, ng)
    col_p <- sprintf("%s_ctx%d_pron",          mc, ng)
    col_b <- sprintf("%s_ctx%d_pron_baseline", mc, ng)

    rt <- load_rt_data[[ds]]()
    df <- scl(merge(rt, preds_spill[[corp]], by = c("story", "zone"), sort = FALSE) %>% ungroup() %>% drop_na())

    set.seed(1)
    folds <- sample(rep(seq_len(n_folds), length.out = nrow(df)))
    d1 <- numeric(nrow(df))
    d2 <- numeric(nrow(df))

    for (k in seq_len(n_folds)) {
        tr <- which(folds != k)
        te <- which(folds == k)
        y  <- df[[outcome]][te]

        lp <- function(col) {
            m <- lm(surp_f(col), data = df[tr, ])
            dnorm(y, predict(m, df[te, ]), sigma(m), log = TRUE)
        }
        lp_o <- lp(col_o)
        lp_p <- lp(col_p)
        lp_b <- lp(col_b)

        d1[te] <- lp_p - lp_o  # negative -> pron worse than orig
        d2[te] <- lp_p - lp_b  # negative -> pron worse than pron_bl
    }

    t1 <- t.test(d1, mu = 0)
    t2 <- t.test(d2, mu = 0)
    cat(sprintf("  %-10s  %5d  %9.3f  %10.2e    %9.3f  %10.2e  (%d)\n",
                ds, ctx, t1$statistic, t1$p.value,
                t2$statistic, t2$p.value, nrow(df)))
}
