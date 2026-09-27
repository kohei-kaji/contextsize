library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)

CANONICAL_ORDER <- c("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
MODEL_COLORS    <- setNames(c("#FF4B00", "#005AFF", "#03AF7A", "#4DC4FF"), CANONICAL_ORDER)
MODEL_LABELS    <- setNames(sub("^gpt2$", "gpt2-small", CANONICAL_ORDER), CANONICAL_ORDER)

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
CORPUS_LABELS <- c(brown = "Brown", ns = "Natural Stories", os = "OneStop", provo = "Provo")

dir.create("../result/png", showWarnings = FALSE, recursive = TRUE)
dir.create("../result/pdf", showWarnings = FALSE, recursive = TRUE)

sanity_df    <- read.csv("../result/sanity_check.csv")
results      <- read.csv("../result/lm_dll.csv")
results_pron <- read.csv("../result/lm_dll_pronominalized.csv")

models_in_data <- unique(results$model)
model_order    <- c(CANONICAL_ORDER[CANONICAL_ORDER %in% models_in_data], sort(setdiff(models_in_data, CANONICAL_ORDER)))
model_colors   <- MODEL_COLORS[model_order]
model_labels   <- MODEL_LABELS[model_order]

results      <- results      %>% mutate(model = factor(model, levels = model_order))
results_pron <- results_pron %>% mutate(model = factor(model, levels = model_order))
sanity_df    <- sanity_df    %>% mutate(model = factor(model, levels = model_order), corpus = factor(corpus, levels = names(CORPUS_LABELS)))

PROVO_DS      <- c("provoff", "provogd", "provotf")
PROVO_MAX_CTX <- 100L
results      <- results      %>% filter(!(dataset %in% PROVO_DS & context > PROVO_MAX_CTX))
results_pron <- results_pron %>% filter(!(dataset %in% PROVO_DS & context > PROVO_MAX_CTX))
sanity_df    <- sanity_df    %>% filter(!(corpus  == "provo"    & context > PROVO_MAX_CTX))

base_theme <- theme_bw() +
    theme(axis.text.x    = element_text(angle = 45, hjust = 1),
          legend.position = "bottom")

STRIP_TEXT   <- element_text(face = "bold", size = 16)
AXIS_TITLE_X <- element_text(size = 20, margin = margin(t = 10))
AXIS_TITLE_Y <- element_text(size = 20, margin = margin(r = 10))
LEGEND_TEXT  <- element_text(size = 18)

save_plot <- function(p, stem, width, height) {
    ggsave(sprintf("../result/png/%s.png", stem), p, width = width, height = height, dpi = 300)
    ggsave(sprintf("../result/pdf/%s.pdf", stem), p, width = width, height = height)
    message(sprintf("Saved: ../result/png/%s.png + .pdf", stem))
}


make_sanity_plot <- function(df, y, ylab, ylo = NULL, yhi = NULL) {
    p <- ggplot(df %>% mutate(context = factor(context, levels = sort(unique(context)))), aes(x = context, y = .data[[y]], color = model, group = model))
    if (!is.null(ylo) && !is.null(yhi))
        p <- p + geom_errorbar(aes(ymin = .data[[ylo]], ymax = .data[[yhi]]), width = 0.3, linewidth = 0.35, alpha = 0.5)
    p + geom_line(linewidth = 0.6, alpha = 0.7, show.legend = FALSE) +
        geom_point(size = 2) +
        facet_wrap(~ corpus, ncol = 2, labeller = as_labeller(CORPUS_LABELS)) +
        scale_x_discrete(name = "Context Window (tokens)") +
        scale_color_manual(values = model_colors, labels = model_labels) +
        guides(color = guide_legend(override.aes = list(size = 4))) +
        ylab(ylab) + labs(color = NULL) +
        base_theme +
        theme(strip.text   = STRIP_TEXT,
              axis.title.x = AXIS_TITLE_X,
              axis.title.y = AXIS_TITLE_Y,
              legend.text  = LEGEND_TEXT)
}

save_plot(make_sanity_plot(sanity_df, "cor_with_unisurp", "Pearson r with unigram surprisal"), "unigram_cor", width = 12, height = 8)
save_plot(make_sanity_plot(sanity_df, "mean_surp", "Mean Surprisal (bits)", "lower_surp", "upper_surp"), "mean_surp", width = 12, height = 8)


# Delta log-likelihood — original texts, w/o and w/ unisurp baseline
context_order <- sort(unique(results$context))

plot_long <- results %>%
    mutate(context = factor(context, levels = context_order), dataset = factor(dataset, levels = names(DS_LABELS))) %>%
    pivot_longer(c(mean_dll_no_unisurp, mean_dll_with_unisurp),
                 names_to  = "baseline_type",
                 values_to = "mean_dll") %>%
    mutate(
        lower_ci = if_else(baseline_type == "mean_dll_no_unisurp", lower_ci_no_unisurp, lower_ci_with_unisurp),
        upper_ci = if_else(baseline_type == "mean_dll_no_unisurp", upper_ci_no_unisurp, upper_ci_with_unisurp),
        baseline_label = if_else(baseline_type == "mean_dll_no_unisurp", "w/o unigram surp", "w/ unigram surp")
    )

make_dll_plot <- function(df) {
    ggplot(df, aes(x = context, y = mean_dll, color = model, group = model)) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "gray50", alpha = 1, linewidth = 0.8) +
        geom_line(linewidth = 0.6, alpha = 0.7, show.legend = FALSE) +
        geom_point(size = 1.5) +
        geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), width = 0.3, linewidth = 0.4, alpha = 0.7, show.legend = FALSE) +
        facet_wrap(~ dataset_label, nrow = 3, scales = "free_y") +
        scale_x_discrete(name = "Context Window (tokens)") +
        scale_color_manual(values = model_colors, labels = model_labels) +
        labs(color = NULL) +
        guides(color = guide_legend(override.aes = list(size = 4))) +
        base_theme +
        theme(strip.text   = STRIP_TEXT,
              axis.title.x = AXIS_TITLE_X,
              axis.title.y = AXIS_TITLE_Y,
              legend.text  = LEGEND_TEXT)
}

p_stems <- list(
    "w/o unigram surp" = "lm_dll_no_unisurp",
    "w/ unigram surp"  = "lm_dll_with_unisurp"
)

target_datasets_for_zoom <- c("Brown SPR", "OneStop FF", "Provo FF")

for (bl in names(p_stems)) {
    sub <- plot_long %>%
        filter(baseline_label == bl) %>%
        mutate(dataset_label = factor(DS_LABELS[as.character(dataset)], levels = unname(DS_LABELS)))

    p_normal <- make_dll_plot(sub) +
        scale_x_discrete(name = "Context Window (tokens)") +
        ylab(NULL) +
        theme(axis.title.y = element_blank(),
              axis.title.x = element_text(size = 18, margin = margin(t = 10), hjust = 0.4))

    zoom_limits <- head(levels(sub$context), 11)

    sub_zoomed <- sub %>%
        filter(dataset_label %in% target_datasets_for_zoom,
               context %in% zoom_limits) %>%
        mutate(dataset_label = paste0(dataset_label, " [Zoom 1-20]"),
               dataset_label = factor(dataset_label, levels = unique(dataset_label)))

    p_zoomed <- make_dll_plot(sub_zoomed) +
        scale_x_discrete(limits = zoom_limits) +
        ylab("Delta Log Likelihood (average per word)") +
        theme(axis.title.x = element_blank())

    p_sep <- ggplot() +
        geom_vline(xintercept = 0, linetype = "dotted", color = "black", linewidth = 1) +
        theme_void()

    combined_plot <- p_zoomed + p_sep + p_normal +
        plot_layout(widths = c(0.6, 0.05, 3), guides = "collect") &
        theme(legend.position = "bottom")

    save_plot(combined_plot, p_stems[[bl]], width = 18, height = 9)
}

# Original vs CorefDisrupted vs SingletonDisrupted DLL curves
ctx_levels <- c("Original", "CorefDisrupted", "SingletonDisrupted")
ctx_shapes <- c("Original" = 16L, "CorefDisrupted" = 17L, "SingletonDisrupted" = 15L)

plot_pron_df <- results_pron %>%
    pivot_longer(c(mean_dll_orig, mean_dll_pron, mean_dll_pron_baseline),
                 names_to  = "ctx_type",
                 values_to = "mean_dll") %>%
    mutate(
        lower_ci = case_when(
            ctx_type == "mean_dll_orig"          ~ lower_orig,
            ctx_type == "mean_dll_pron"          ~ lower_pron,
            ctx_type == "mean_dll_pron_baseline" ~ lower_pron_baseline
        ),
        upper_ci = case_when(
            ctx_type == "mean_dll_orig"          ~ upper_orig,
            ctx_type == "mean_dll_pron"          ~ upper_pron,
            ctx_type == "mean_dll_pron_baseline" ~ upper_pron_baseline
        ),
        ctx_label = case_when(
            ctx_type == "mean_dll_orig"          ~ "Original",
            ctx_type == "mean_dll_pron"          ~ "CorefDisrupted",
            ctx_type == "mean_dll_pron_baseline" ~ "SingletonDisrupted"
        ),
        ctx_label     = factor(ctx_label, levels = ctx_levels),
        context       = factor(context, levels = sort(unique(context))),
        model         = factor(model, levels = model_order),
        dataset_label = factor(DS_LABELS[dataset], levels = unname(DS_LABELS))
    ) %>%
    filter(!is.na(mean_dll))

p_pron <- ggplot(plot_pron_df,
                 aes(x = context, y = mean_dll, color = model,
                     shape = ctx_label, group = interaction(model, ctx_label))) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50", alpha = 1, linewidth = 0.8) +
    geom_line(linewidth = 0.6, alpha = 0.8, show.legend = FALSE) +
    geom_point(size = 2) +
    geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci),
                  width = 0.3, linewidth = 0.4, alpha = 0.6, show.legend = FALSE) +
    facet_wrap(~ dataset_label, nrow = 3, scales = "free_y") +
    scale_x_discrete(name = "Context Window (tokens)") +
    scale_color_manual(values = model_colors, labels = model_labels) +
    scale_shape_manual(values = ctx_shapes, name = "Context type") +
    ylab("Delta Log Likelihood (average per word)") +
    labs(color = NULL) +
    guides(color = guide_legend(override.aes = list(size = 4)),
           shape = guide_legend(override.aes = list(size = 4))) +
    base_theme +
    theme(strip.text   = STRIP_TEXT,
          axis.title.x = AXIS_TITLE_X,
          axis.title.y = AXIS_TITLE_Y,
          legend.text  = LEGEND_TEXT)

save_plot(p_pron, "dll_pron_comparison", width = 12, height = 8)


# Delta DLL — decrease when using original vs CorefDisrupted/SingletonDisrupted
delta_long <- results_pron %>%
    pivot_longer(
        cols      = c(mean_dll_delta, mean_dll_delta_baseline),
        names_to  = "delta_type",
        values_to = "raw_delta"
    ) %>%
    mutate(
        raw_lower   = if_else(delta_type == "mean_dll_delta", lower_delta,  lower_delta_baseline),
        raw_upper   = if_else(delta_type == "mean_dll_delta", upper_delta,  upper_delta_baseline),
        mean_dll    = -raw_delta,
        lower_ci    = -raw_upper,
        upper_ci    = -raw_lower,
        delta_label = factor(
            if_else(delta_type == "mean_dll_delta", "vs CorefDisrupted", "vs SingletonDisrupted"),
            levels = c("vs CorefDisrupted", "vs SingletonDisrupted")
        ),
        context       = factor(context, levels = sort(unique(context))),
        model         = factor(model, levels = model_order),
        dataset_label = factor(DS_LABELS[dataset], levels = unname(DS_LABELS))
    ) %>%
    filter(!is.na(mean_dll))

p_delta <- ggplot(delta_long,
                  aes(x = context, y = mean_dll, color = model, shape = delta_label, group = interaction(model, delta_label))) +
    geom_hline(yintercept = 0, linetype = "dotted", color = "gray50") +
    geom_line(linewidth = 0.6, alpha = 0.7, show.legend = FALSE) +
    geom_point(size = 2) +
    geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), width = 0.3, linewidth = 0.4, alpha = 0.6, show.legend = FALSE) +
    facet_wrap(~ dataset_label, nrow = 3, scales = "free_y") +
    scale_x_discrete(name = "Context Window (tokens)") +
    scale_color_manual(values = model_colors, labels = model_labels) +
    scale_shape_manual(
        values = c("vs CorefDisrupted" = 16L, "vs SingletonDisrupted" = 17L),
        name   = NULL
    ) +
    ylab("Decrease in Log Likelihood by Context Destruction") +
    labs(color = NULL) +
    guides(color = guide_legend(override.aes = list(size = 4)),
           shape = guide_legend(override.aes = list(size = 4))) +
    base_theme +
    theme(strip.text   = STRIP_TEXT,
          axis.title.x = AXIS_TITLE_X,
          axis.title.y = AXIS_TITLE_Y,
          legend.text  = LEGEND_TEXT)

save_plot(p_delta, "dll_pron_delta", width = 16, height = 8)



DELTA_COLORS <- c("vs CorefDisrupted" = "#D55E00", "vs SingletonDisrupted" = "#005A9C")

for (mdl in model_order) {
    mdl_stem <- gsub("-", "_", mdl)

    for (bl in names(p_stems)) {
        sub <- plot_long %>%
            filter(baseline_label == bl, model == mdl) %>%
            mutate(dataset_label = factor(DS_LABELS[as.character(dataset)], levels = unname(DS_LABELS)))
        bl_stem <- gsub(" ", "_", gsub("/", "", bl))
        save_plot(
            make_dll_plot(sub) + ylab("Delta Log Likelihood (average per word)"),
            sprintf("lm_dll_%s_%s", bl_stem, mdl_stem), width = 12, height = 6
        )
    }

    p_delta_mdl <- ggplot(delta_long %>% filter(model == mdl),
                          aes(x = context, y = mean_dll, color = delta_label,
                              shape = delta_label, group = delta_label)) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "gray50", alpha = 1, linewidth = 0.8) +
        geom_line(linewidth = 0.6, alpha = 0.7, show.legend = FALSE) +
        geom_point(size = 2) +
        geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci),
                      width = 0.3, linewidth = 0.4, alpha = 0.6, show.legend = FALSE) +
        facet_wrap(~ dataset_label, nrow = 3, scales = "free_y") +
        scale_x_discrete(name = "Context Window (tokens)") +
        scale_color_manual(values = DELTA_COLORS, name = NULL) +
        scale_shape_manual(
            values = c("vs CorefDisrupted" = 16L, "vs SingletonDisrupted" = 17L),
            name   = NULL
        ) +
        ylab("Decrease in Log Likelihood by Context Destruction") +
        guides(color = guide_legend(override.aes = list(size = 4)),
               shape = guide_legend(override.aes = list(size = 4))) +
        base_theme +
        theme(strip.text   = STRIP_TEXT,
              axis.title.x = AXIS_TITLE_X,
              axis.title.y = AXIS_TITLE_Y,
              legend.text  = LEGEND_TEXT)

    save_plot(p_delta_mdl, sprintf("dll_pron_delta_%s", mdl_stem), width = 18, height = 9)
}

# Mean surprisal distributions of Original, CorefDisrupted, and SingletonDisrupted
surp_base <- "../data/surp"

type_dirs <- list(
    ns    = c(orig = "ns",    pron = "ns_pronominalized",      bl = "ns_baseline_pronominalized"),
    brown = c(orig = "brown", pron = "brown_pronominalized",   bl = "brown_baseline_pronominalized"),
    os    = c(orig = "os",    pron = "onestop_pronominalized", bl = "onestop_baseline_pronominalized"),
    provo = c(orig = "provo", pron = "provo_pronominalized",   bl = "provo_baseline_pronominalized")
)

surp_rows <- list()
for (corp in names(type_dirs)) {
    for (type_name in names(type_dirs[[corp]])) {
        dir_path <- file.path(surp_base, type_dirs[[corp]][[type_name]])
        for (m in model_order) {
            mdir <- file.path(dir_path, m)
            if (!dir.exists(mdir)) next
            files <- list.files(mdir, pattern = "^context_\\d+\\.txt$", full.names = FALSE)
            for (f in files) {
                ng   <- as.integer(gsub("context_|\\.txt", "", f))
                ctx  <- ng - 1L
                vals <- scan(file.path(mdir, f), what = numeric(), quiet = TRUE)
                if (length(vals) == 0) next
                n  <- length(vals)
                mn <- mean(vals)
                se <- sd(vals) / sqrt(n)
                surp_rows[[length(surp_rows) + 1]] <- data.frame(
                    corpus = corp, type = type_name, model = m, context = ctx,
                    mean_surp  = mn,
                    lower_surp = mn - 1.96 * se,
                    upper_surp = mn + 1.96 * se,
                    stringsAsFactors = FALSE
                )
            }
        }
    }
}

surp_dist_df <- bind_rows(surp_rows) %>%
    filter(!(corpus == "provo" & context > PROVO_MAX_CTX)) %>%
    mutate(
        model   = factor(model,   levels = model_order),
        corpus  = factor(corpus,  levels = names(CORPUS_LABELS)),
        type    = factor(type,    levels = c("orig", "pron", "bl")),
        context = factor(context, levels = sort(unique(context)))
    )

TYPE_LABELS <- c(orig = "Original", pron = "CorefDisrupted", bl = "SingletonDisrupted")
TYPE_SHAPES <- c(orig = 16L, pron = 17L, bl = 15L)

p_surp_dist <- ggplot(surp_dist_df,
    aes(x = context, y = mean_surp, color = model, shape = type, group = interaction(model, type))) +
    geom_line(linewidth = 0.6, alpha = 0.7, show.legend = FALSE) +
    geom_point(size = 2) +
    geom_errorbar(aes(ymin = lower_surp, ymax = upper_surp), width = 0.3, linewidth = 0.35, alpha = 0.5, show.legend = FALSE) +
    facet_wrap(~ corpus, ncol = 2, labeller = as_labeller(CORPUS_LABELS)) +
    scale_x_discrete(name = "Context Window (tokens)") +
    scale_color_manual(values = model_colors, labels = model_labels, name = "Model") +
    scale_shape_manual(values = TYPE_SHAPES, labels = TYPE_LABELS, name = "Context Type") +
    ylab("Mean Surprisal (bits)") +
    labs(color = NULL) +
    guides(color = guide_legend(override.aes = list(size = 4)),
           shape = guide_legend(override.aes = list(size = 4))) +
    base_theme +
    theme(strip.text   = STRIP_TEXT,
          axis.title.x = AXIS_TITLE_X,
          axis.title.y = AXIS_TITLE_Y,
          legend.text  = LEGEND_TEXT)

save_plot(p_surp_dist, "surp_dist_comparison", width = 18, height = 9)
