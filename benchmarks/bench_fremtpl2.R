# =============================================================================
# Benchmark: rustyglm vs glm() base R — con soporte offset y benchmark paralelo
# Dataset: freMTPL2freq (~678k pólizas francesas)
# =============================================================================

library(rustyglm)
library(CASdatasets)
library(bench)

data(freMTPL2freq)

# --- Preparar datos ---
dat <- freMTPL2freq
dat$ClaimNb  <- as.integer(dat$ClaimNb)
dat$Exposure <- as.double(dat$Exposure)
dat$Area     <- as.factor(dat$Area)
dat$VehGas   <- as.factor(dat$VehGas)

cat("Dataset: freMTPL2freq\n")
cat(sprintf("Filas: %d | Columnas: %d\n\n", nrow(dat), ncol(dat)))

formula_freq <- ClaimNb ~ VehAge + DrivAge + BonusMalus + VehPower +
  Area + VehGas + Density

# =============================================================================
# PARTE 1 — Benchmark de velocidad y memoria por tamaño de muestra
# =============================================================================
cat("=== PARTE 1: Velocidad y Memoria ===\n\n")

sizes <- c("10k" = 10000, "50k" = 50000, "100k" = 100000, "678k (full)" = nrow(dat))

bench_results <- lapply(names(sizes), function(nm) {
  n <- sizes[[nm]]
  d <- dat[seq_len(n), ]
  off <- log(d$Exposure)
  
  cat(sprintf("--- n = %s ---\n", nm))
  bm <- bench::mark(
    rustyglm = rustyglm(formula_freq, data = d, family = "poisson", offset = off),
    glm_base = glm(formula_freq, data = d, family = poisson(), offset = log(Exposure)),
    iterations = 5,
    check = FALSE
  )
  print(bm[, c("expression", "min", "median", "mem_alloc", "n_itr")])
  cat("\n")
  bm
})
names(bench_results) <- names(sizes)

# Tabla resumen
cat("=== RESUMEN: Velocidad y Memoria ===\n")
cat(sprintf("%-14s %10s %10s %10s %10s %8s\n",
            "Muestra", "rust_med", "r_med", "rust_mem", "r_mem", "Speedup"))
cat(rep("-", 68), "\n", sep="")

for (nm in names(bench_results)) {
  bm     <- bench_results[[nm]]
  t_rust <- as.numeric(bm$median[1])
  t_r    <- as.numeric(bm$median[2])
  m_rust <- bm$mem_alloc[1]
  m_r    <- bm$mem_alloc[2]
  cat(sprintf("%-14s %10s %10s %10s %10s %7.2fx\n",
              nm,
              format(bench::as_bench_time(t_rust)),
              format(bench::as_bench_time(t_r)),
              format(m_rust, units = "auto"),
              format(m_r,    units = "auto"),
              t_r / t_rust))
}

# =============================================================================
# PARTE 2 — Validación numérica con offset correcto
# =============================================================================
cat("\n=== PARTE 2: Validación Numérica (full dataset, con offset) ===\n")

mod_rust <- rustyglm(formula_freq, data = dat, family = "poisson",
                     offset = log(dat$Exposure))
mod_r    <- glm(formula_freq, data = dat, family = poisson(),
                offset = log(Exposure))

diff <- abs(coef(mod_rust) - coef(mod_r))
cat(sprintf("Diferencia máxima: %.2e\n", max(diff)))
cat(sprintf("Diferencia media:  %.2e\n", mean(diff)))

comparison <- data.frame(
  rustyglm = round(coef(mod_rust), 6),
  glm_r    = round(coef(mod_r),    6),
  diff     = round(diff,           8)
)
print(comparison)

# =============================================================================
# PARTE 3 — Benchmark paralelo: 1 hilo vs N hilos (rayon)
# =============================================================================
cat("\n=== PARTE 3: Benchmark Paralelo (rayon threads) ===\n")
cat("RustyStats usa rayon internamente. Controlamos threads via RAYON_NUM_THREADS.\n\n")

# Usamos el full dataset para que el paralelismo sea visible
d_full <- dat
off_full <- log(d_full$Exposure)

# Número de cores disponibles
n_cores <- parallel::detectCores()
cat(sprintf("Cores disponibles: %d\n\n", n_cores))

thread_counts <- unique(c(1, 2, 4, min(8, n_cores), n_cores))
thread_counts <- sort(thread_counts[thread_counts <= n_cores])

parallel_results <- lapply(thread_counts, function(nt) {
  # rayon respeta RAYON_NUM_THREADS antes de crear el thread pool
  Sys.setenv(RAYON_NUM_THREADS = as.character(nt))
  
  cat(sprintf("Threads: %d\n", nt))
  bm <- bench::mark(
    rustyglm(formula_freq, data = d_full, family = "poisson", offset = off_full),
    iterations = 5,
    check = FALSE
  )
  cat(sprintf("  mediana: %s\n\n", format(bm$median)))
  list(threads = nt, median = as.numeric(bm$median), bm = bm)
})

# Restaurar default
Sys.setenv(RAYON_NUM_THREADS = "")

# Tabla paralelo
cat("=== TABLA PARALELO ===\n")
cat(sprintf("%-10s %12s %10s\n", "Threads", "Mediana", "Speedup vs 1"))
cat(rep("-", 36), "\n", sep="")

t_single <- parallel_results[[1]]$median
for (pr in parallel_results) {
  cat(sprintf("%-10d %12s %9.2fx\n",
              pr$threads,
              format(bench::as_bench_time(pr$median)),
              t_single / pr$median))
}

cat("\n=== glm() base R (referencia, 1 thread) ===\n")
bm_r_ref <- bench::mark(
  glm(formula_freq, data = d_full, family = poisson(), offset = log(Exposure)),
  iterations = 3, check = FALSE
)
cat(sprintf("glm() mediana: %s\n", format(bm_r_ref$median)))
cat(sprintf("rustyglm 1T vs glm(): %.2fx\n",
            as.numeric(bm_r_ref$median) / t_single))
cat(sprintf("rustyglm %dT vs glm(): %.2fx\n",
            n_cores,
            as.numeric(bm_r_ref$median) / parallel_results[[length(parallel_results)]]$median))