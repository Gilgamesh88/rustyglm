# =============================================================================
# Benchmark paralelo REAL: fitting N modelos simultáneos
# Caso de uso actuarial: N territorios/regiones independientes
# =============================================================================

library(rustyglm)
library(CASdatasets)
library(bench)

data(freMTPL2freq)

dat <- freMTPL2freq
dat$Exposure <- as.double(dat$Exposure)
dat$Area     <- as.factor(dat$Area)
dat$VehGas   <- as.factor(dat$VehGas)

formula_freq <- ClaimNb ~ VehAge + DrivAge + BonusMalus + VehPower + VehGas + Density

# Simulamos N "regiones" como subconjuntos independientes del dataset
# (caso real: modelo por territorio, por línea de negocio, CV folds)
regiones <- split(dat, dat$Region)
n_regiones <- length(regiones)
cat(sprintf("Regiones disponibles: %d\n", n_regiones))
cat(sprintf("Tamaño promedio por región: %.0f filas\n\n",
            mean(sapply(regiones, nrow))))

# =============================================================================
# PARTE A — Sequential vs Parallel: ajustar todas las regiones
# =============================================================================
cat("=== A: Sequential vs Parallel — ajuste de", n_regiones, "regiones ===\n\n")

# Función que ajusta un modelo por región
fit_region <- function(d) {
  off <- log(d$Exposure)
  tryCatch(
    rustyglm(formula_freq, data = d, family = "poisson", offset = off),
    error = function(e) NULL
  )
}

fit_region_glm <- function(d) {
  tryCatch(
    glm(formula_freq, data = d, family = poisson(), offset = log(Exposure)),
    error = function(e) NULL
  )
}

bm_seq <- bench::mark(
  # Sequential rustyglm
  rust_seq = lapply(regiones, fit_region),
  
  # Parallel rustyglm (parallel::mclapply en Linux; parLapply en Windows)
  rust_par = {
    cl <- parallel::makeCluster(min(n_regiones, parallel::detectCores()))
    parallel::clusterEvalQ(cl, { library(rustyglm) })
    res <- parallel::parLapply(cl, regiones, fit_region)
    parallel::stopCluster(cl)
    res
  },
  
  # Sequential glm base R
  glm_seq = lapply(regiones, fit_region_glm),
  
  iterations = 3,
  check = FALSE
)

print(bm_seq[, c("expression", "min", "median", "mem_alloc", "n_itr")])

# =============================================================================
# PARTE B — Cross-validation: 5 folds en full dataset
# =============================================================================
cat("\n=== B: Cross-Validation 5-fold (sequential vs parallel) ===\n\n")

set.seed(42)
n      <- nrow(dat)
folds  <- sample(rep(1:5, length.out = n))

fit_fold <- function(k) {
  train <- dat[folds != k, ]
  off   <- log(train$Exposure)
  rustyglm(formula_freq, data = train, family = "poisson", offset = off)
}

fit_fold_glm <- function(k) {
  train <- dat[folds != k, ]
  glm(formula_freq, data = train, family = poisson(), offset = log(Exposure))
}

bm_cv <- bench::mark(
  rust_cv_seq = lapply(1:5, fit_fold),
  
  rust_cv_par = {
    cl <- parallel::makeCluster(5)
    parallel::clusterEvalQ(cl, { library(rustyglm) })
    parallel::clusterExport(cl, c("dat", "folds", "formula_freq"), envir = environment())
    res <- parallel::parLapply(cl, 1:5, fit_fold)
    parallel::stopCluster(cl)
    res
  },
  
  glm_cv_seq = lapply(1:5, fit_fold_glm),
  
  iterations = 3,
  check = FALSE
)

print(bm_cv[, c("expression", "min", "median", "mem_alloc", "n_itr")])

# =============================================================================
# PARTE C — Regularización: donde rayon SÍ paralela internamente
# =============================================================================
cat("\n=== C: Memoria total — modelo único full dataset ===\n")
cat("(El resultado realmente impresionante de la sesión anterior)\n\n")

cat(sprintf("%-20s %12s %12s %10s\n", "Método", "Mediana", "Memoria", "Nota"))
cat(rep("-", 58), "\n", sep="")

# rustyglm full
bm_full_rust <- bench::mark(
  rustyglm(formula_freq, data = dat, family = "poisson", offset = log(dat$Exposure)),
  iterations = 3, check = FALSE
)

# glm full  
bm_full_r <- bench::mark(
  glm(formula_freq, data = dat, family = poisson(), offset = log(Exposure)),
  iterations = 3, check = FALSE
)

cat(sprintf("%-20s %12s %12s %10s\n",
            "rustyglm (678k)",
            format(bm_full_rust$median),
            format(bm_full_rust$mem_alloc, units = "auto"),
            "Rust IRLS"))

cat(sprintf("%-20s %12s %12s %10s\n",
            "glm() base R (678k)",
            format(bm_full_r$median),
            format(bm_full_r$mem_alloc, units = "auto"),
            "R IRLS"))

mem_ratio <- as.numeric(bm_full_r$mem_alloc) / as.numeric(bm_full_rust$mem_alloc)
cat(sprintf("\nReduccion de memoria: %.1fx menos que glm() base R\n", mem_ratio))

# =============================================================================
# RESUMEN EJECUTIVO
# =============================================================================
cat("\n", rep("=", 58), "\n", sep="")
cat("RESUMEN EJECUTIVO\n")
cat(rep("=", 58), "\n", sep="")

t_rust_seq <- as.numeric(bm_seq$median[bm_seq$expression == "rust_seq"])
t_rust_par <- as.numeric(bm_seq$median[bm_seq$expression == "rust_par"])
t_glm_seq  <- as.numeric(bm_seq$median[bm_seq$expression == "glm_seq"])

t_cv_rust_seq <- as.numeric(bm_cv$median[bm_cv$expression == "rust_cv_seq"])
t_cv_rust_par <- as.numeric(bm_cv$median[bm_cv$expression == "rust_cv_par"])
t_cv_glm_seq  <- as.numeric(bm_cv$median[bm_cv$expression == "glm_cv_seq"])

cat(sprintf("\n[Regiones] rust_seq vs glm_seq:  %.2fx\n", t_glm_seq / t_rust_seq))
cat(sprintf("[Regiones] rust_par vs rust_seq:  %.2fx\n", t_rust_seq / t_rust_par))
cat(sprintf("[Regiones] rust_par vs glm_seq:   %.2fx\n", t_glm_seq / t_rust_par))

cat(sprintf("\n[CV 5-fold] rust_seq vs glm_seq:  %.2fx\n", t_cv_glm_seq / t_cv_rust_seq))
cat(sprintf("[CV 5-fold] rust_par vs rust_seq:  %.2fx\n", t_cv_rust_seq / t_cv_rust_par))
cat(sprintf("[CV 5-fold] rust_par vs glm_seq:   %.2fx\n", t_cv_glm_seq / t_cv_rust_par))

cat(sprintf("\n[Memoria]  rustyglm usa %.1fx menos RAM que glm() (678k obs)\n", mem_ratio))
cat(sprintf("           glm(): %.1fGB | rustyglm: %.0fMB\n",
            as.numeric(bm_full_r$mem_alloc) / 1e9,
            as.numeric(bm_full_rust$mem_alloc) / 1e6))