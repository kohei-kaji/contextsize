library(readr)
library(dplyr)
library(brms)

dat <- read_csv("../result/lm_dll.csv", show_col_types = FALSE) %>% mutate(dll = mean_dll_with_unisurp)


priors <- c(
  set_prior("normal(0, 0.1)", class = "b"),
  set_prior("normal(0, 5)", class = "Intercept"),
  set_prior("exponential(1)", class = "sd"),
  set_prior("exponential(1)", class = "sigma")
)
fit <- brm(
    dll ~ log(context) + (1 | model) + (1 | dataset),
    data   = dat,
    family = gaussian(),
    prior = priors,
    chains = 4,
    iter   = 2000,
    warmup = 1000,
    cores  = 4,
    seed   = 42,
    control = list(adapt_delta = 0.99, max_treedepth = 15),
    file   = "../result/brms_dll_fit"
)

print(summary(fit))

write_csv(as.data.frame(fixef(fit)), "../result/brms_dll_fixef.csv")
