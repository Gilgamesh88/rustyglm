# =============================================================================
# Benchmark: rustyglm SMuRF (Rust) vs smurf R package vs glmnet vs glm()
# Dataset: freMTPL2freq — 678k pólizas francesas de seguro de auto
# =============================================================================
# Variables y penalizaciones actuarialmente motivadas:
#   VehAge   (1-15, ordinal)  -> flasso: funde edades de vehículo contiguas
#   DrivAge  (continuo)       -> lasso:  shrinkage estándar
#   VehPower (ordinal)        -> flasso: funde potencias contiguas
#   Area     (A<B<C<D<E<F)   -> flasso: funde zonas geográficas
#   VehGas   (Diesel/Regular) -> grouplasso: elimina o mantiene el factor
#   Density  (continuo)       -> lasso
#
# NOTA: BonusMalus (180+ niveles) se excluye del SMuRF para comparación justa
# con el paquete R smurf, que no soporta bien matrices ADMM de esa dimensión.
# =============================================================================

library(rustyglm)
library(CASdatasets)
library(smurf)
library(bench)

# =============================================================================
# 0. Cargar y preparar datos
# =============================================================================
cat("Cargando freMTPL2freq...\n")
data(freMTPL2freq)

dat          <- freMTPL2freq
dat$Exposure <- as.double(dat$Exposure)

# Factores para variables ordinales y nominales
dat$VehAge_f   <- factor(dat$VehAge)
dat$VehPower_f <- factor(dat$VehPower)
dat$Area_f     <- factor(dat$Area, levels = c("A","B","C","D","E","F"))
dat$VehGas_f   <- factor(dat$VehGas)

LAMBDA <- 0.005
cat(sprintf("Filas totales: %d | Lambda: %s\n\n", nrow(dat), LAMBDA))

# Fórmulas compartidas para comparación justa
# (mismas variables en SMuRF Rust y smurf R)
formula_smurf <- ClaimNb ~ VehAge_f + DrivAge + VehPower_f + Area_f + VehGas_f + Density

formula_smurf_r <- ClaimNb ~ p(VehAge_f,   pen = "flasso") +
                               p(DrivAge,    pen = "lasso") +
                               p(VehPower_f, pen = "flasso") +
                               p(Area_f,     pen = "flasso") +
                               p(VehGas_f,   pen = "grouplasso") +
                               p(Density,    pen = "lasso")

pen_types_rust <- list(
  VehAge_f   = "flasso",
  DrivAge    = "lasso",
  VehPower_f = "flasso",
  Area_f     = "flasso",
  VehGas_f   = "grouplasso",
  Density    = "lasso"
)

# Fórmula para rustyglm estándar (variables numéricas, sin factorizar)
formula_base <- ClaimNb ~ VehAge + DrivAge + BonusMalus + VehPower +
                           Area + VehGas + Density

# =============================================================================
# Función auxiliar: preparar submuestra con niveles limpios
# =============================================================================
prep_sample <- function(data, indices) {
  d            <- data[indices, ]
  d$VehAge_f   <- droplevels(d$VehAge_f)
  d$VehPower_f <- droplevels(d$VehPower_f)
  d$Area_f     <- droplevels(factor(d$Area_f))   # factor simple, sin ordered
  d$VehGas_f   <- droplevels(d$VehGas_f)
  d$off        <- log(d$Exposure)
  d
}

# =============================================================================
# PARTE 1 — Ajuste en n=50k: rustyglm SMuRF vs smurf R vs baselines
# =============================================================================
cat("=== PARTE 1: Ajuste en n=50k ===\n\n")

set.seed(42)
d50 <- prep_sample(dat, sample(nrow(dat), 50000))

# --- 1a. rustyglm SMuRF (Rust FISTA) ---
cat("Ajustando rustyglm SMuRF (Rust)...\n")
t_rust_smurf <- system.time({
  mod_rust_smurf <- rustyglm_smurf(
    formula   = formula_smurf,
    data      = d50,
    family    = "poisson",
    offset    = d50$off,
    pen.types = pen_types_rust,
    lambda    = LAMBDA,
    maxiter   = 2000L,
    epsilon   = 1e-5
  )
})
cat(sprintf("  OK — %d iteraciones, convergió: %s\n",
            mod_rust_smurf$iterations, mod_rust_smurf$converged))

# --- 1b. smurf R package ---
cat("Ajustando smurf (R)...\n")
t_r_smurf <- system.time({
  mod_r_smurf <- tryCatch(
    glmsmurf(
      formula     = formula_smurf_r,
      family      = poisson(),
      data        = d50,
      offset      = d50$off,
      lambda      = LAMBDA,
      pen.weights = "glm.stand"
    ),
    error = function(e) {
      cat("  ERROR smurf R:", conditionMessage(e), "\n")
      NULL
    }
  )
})
if (!is.null(mod_r_smurf)) cat("  OK\n")

# --- 1c. rustyglm Lasso (baseline regularizado) ---
cat("Ajustando rustyglm Lasso...\n")
t_rust_lasso <- system.time({
  mod_rust_lasso <- rustyglm(
    formula        = formula_base,
    data           = d50,
    family         = "poisson",
    offset         = d50$off,
    regularization = "lasso",
    lambda         = LAMBDA
  )
})

# --- 1d. glm() base R (sin regularización) ---
cat("Ajustando glm() base R...\n")
t_glm <- system.time({
  mod_glm <- glm(
    formula = formula_base,
    data    = d50,
    family  = poisson(),
    offset  = log(Exposure)
  )
})

# Tabla de tiempos
cat("\n--- Tiempos de ajuste (n=50k) ---\n")
cat(sprintf("  rustyglm SMuRF (Rust): %.2fs\n", t_rust_smurf["elapsed"]))
if (!is.null(mod_r_smurf))
  cat(sprintf("  smurf R:               %.2fs\n", t_r_smurf["elapsed"]))
cat(sprintf("  rustyglm Lasso:        %.2fs\n", t_rust_lasso["elapsed"]))
cat(sprintf("  glm() base R:          %.2fs\n", t_glm["elapsed"]))
if (!is.null(mod_r_smurf))
  cat(sprintf("  Speedup SMuRF Rust vs smurf R: %.1fx\n",
              t_r_smurf["elapsed"] / t_rust_smurf["elapsed"]))

# =============================================================================
# PARTE 2 — Inspección de niveles fusionados (ventaja actuarial clave de SMuRF)
# =============================================================================
cat("\n=== PARTE 2: Fusión de niveles (Fused Lasso) ===\n\n")

coef_smurf_rust <- coef(mod_rust_smurf)

# Area
cat("--- Area (flasso): rustyglm SMuRF ---\n")
area_rust <- coef_smurf_rust[grep("Area", names(coef_smurf_rust))]
print(round(area_rust, 5))
n_unique_area <- length(unique(round(area_rust, 4)))
cat(sprintf("  Niveles únicos: %d de %d (fusionados: %d)\n\n",
            n_unique_area, length(area_rust),
            length(area_rust) - n_unique_area))

cat("--- Area: glm() base R (sin fusión) ---\n")
area_glm <- coef(mod_glm)[grep("Area", names(coef(mod_glm)))]
print(round(area_glm, 5))

# VehGas
cat("\n--- VehGas (grouplasso): rustyglm SMuRF ---\n")
vgas_rust <- coef_smurf_rust[grep("VehGas", names(coef_smurf_rust))]
if (length(vgas_rust) > 0) {
  eliminado <- all(abs(vgas_rust) < 1e-8)
  cat(sprintf("  Coef: %s %s\n",
              paste(round(vgas_rust, 6), collapse = " "),
              if (eliminado) "-> ELIMINADO por Group Lasso" else "-> activo"))
}

# VehAge
cat("\n--- VehAge (flasso): rustyglm SMuRF ---\n")
vage_rust     <- coef_smurf_rust[grep("VehAge", names(coef_smurf_rust))]
n_unique_vage <- length(unique(round(vage_rust, 4)))
cat(sprintf("  Niveles únicos: %d de %d\n", n_unique_vage, length(vage_rust)))

# Comparación con smurf R si funcionó
if (!is.null(mod_r_smurf)) {
  cat("\n--- Comparación Area: smurf R vs rustyglm SMuRF ---\n")
  area_r <- coef(mod_r_smurf)[grep("Area", names(coef(mod_r_smurf)))]
  if (length(area_r) > 0 && length(area_rust) > 0) {
    len <- min(length(area_r), length(area_rust))
    print(data.frame(
      smurf_R    = round(head(area_r,    len), 5),
      rust_smurf = round(head(area_rust, len), 5)
    ))
  }
}

# =============================================================================
# PARTE 3 — Benchmark formal bench::mark (n=20k, 3 iteraciones)
# =============================================================================
cat("\n=== PARTE 3: Benchmark formal (n=20k, 3 iteraciones) ===\n\n")

set.seed(99)
d20 <- prep_sample(dat, sample(nrow(dat), 20000))

bm <- bench::mark(
  `SMuRF Rust` = rustyglm_smurf(
    formula   = formula_smurf,
    data      = d20,
    family    = "poisson",
    offset    = d20$off,
    pen.types = pen_types_rust,
    lambda    = LAMBDA,
    maxiter   = 1000L,
    epsilon   = 1e-5
  ),
  `smurf R` = tryCatch(
    glmsmurf(
      formula     = formula_smurf_r,
      family      = poisson(),
      data        = d20,
      offset      = d20$off,
      lambda      = LAMBDA,
      pen.weights = "glm.stand"
    ),
    error = function(e) NULL
  ),
  `rustyglm Lasso` = rustyglm(
    formula        = formula_base,
    data           = d20,
    family         = "poisson",
    offset         = d20$off,
    regularization = "lasso",
    lambda         = LAMBDA
  ),
  `glm base R` = glm(
    formula = formula_base,
    data    = d20,
    family  = poisson(),
    offset  = log(Exposure)
  ),
  iterations = 3,
  check      = FALSE
)

print(bm[, c("expression", "min", "median", "mem_alloc", "n_itr")])

# =============================================================================
# PARTE 4 — Credibilidad Lasso (modelo nacional -> modelo estatal)
# =============================================================================
cat("\n=== PARTE 4: Credibilidad Lasso ===\n\n")

# Modelo "nacional": full dataset sin regularización
cat("Ajustando modelo nacional (678k obs, prior de credibilidad)...\n")
mod_nacional <- rustyglm(
  formula = formula_base,
  data    = dat,
  family  = "poisson",
  offset  = log(dat$Exposure)
)
cat(sprintf("  Deviance nacional: %.1f\n\n", mod_nacional$deviance))

# Predicciones del modelo nacional sobre el subconjunto de 50k
complement_rates <- predict(mod_nacional, newdata = d50, type = "response")

# Modelo "estatal" — lambda pequeño: más peso al modelo local
cat("Ajustando modelo estatal: credibilidad lambda=0.05...\n")
mod_cred_05 <- rustyglm_credibility(
  formula        = formula_base,
  data           = d50,
  family         = "poisson",
  complement     = complement_rates,
  offset         = d50$off,
  regularization = "lasso",
  lambda         = 0.05
)

# Modelo "estatal" — lambda grande: más peso al prior nacional
cat("Ajustando modelo estatal: credibilidad lambda=0.20...\n")
mod_cred_20 <- rustyglm_credibility(
  formula        = formula_base,
  data           = d50,
  family         = "poisson",
  complement     = complement_rates,
  offset         = d50$off,
  regularization = "lasso",
  lambda         = 0.20
)

# Tabla comparativa de shrinkage
coef_names <- names(coef(mod_rust_lasso))
coef_glm_base <- coef(mod_glm)[coef_names]

comp_df <- data.frame(
  GLM_sin_reg      = round(coef_glm_base,                              5),
  Lasso_std        = round(coef(mod_rust_lasso),                       5),
  Cred_lambda_0.05 = round(coef(mod_cred_05)[coef_names],              5),
  Cred_lambda_0.20 = round(coef(mod_cred_20)[coef_names],              5)
)
cat("\nCoeficientes: GLM vs Lasso estándar vs Credibilidad (dos lambdas)\n")
print(comp_df)
cat("\n(mayor lambda = más peso al modelo nacional; coef más cercanos a 0)\n")

credibility_summary(mod_cred_05)

# =============================================================================
# RESUMEN EJECUTIVO
# =============================================================================
cat("\n", rep("=", 62), "\n", sep = "")
cat("RESUMEN EJECUTIVO\n")
cat(rep("=", 62), "\n\n", sep = "")

if (!is.null(mod_r_smurf)) {
  cat(sprintf("Speedup SMuRF Rust vs smurf R (n=50k):  %.1fx\n",
              t_r_smurf["elapsed"] / t_rust_smurf["elapsed"]))
}
cat(sprintf("Speedup rustyglm vs glm() (n=50k):      %.1fx\n",
            t_glm["elapsed"] / t_rust_lasso["elapsed"]))

rust_mem <- bm$mem_alloc[bm$expression == "SMuRF Rust"]
glm_mem  <- bm$mem_alloc[bm$expression == "glm base R"]
if (length(rust_mem) > 0 && length(glm_mem) > 0)
  cat(sprintf("Memoria: SMuRF Rust %s | glm base R %s\n",
              format(rust_mem, units = "auto"),
              format(glm_mem,  units = "auto")))

cat(sprintf("\nFused Lasso Area   — niveles únicos: %d de %d\n",
            n_unique_area, length(area_rust)))
cat(sprintf("Fused Lasso VehAge — niveles únicos: %d de %d\n",
            n_unique_vage, length(vage_rust)))
cat("(niveles con coef idéntico quedan automáticamente fusionados)\n")
cat("\nCredibilidad Lasso:\n")
cat("  lambda=0.05 -> shrinkage suave hacia modelo nacional\n")
cat("  lambda=0.20 -> mayor peso al prior, coef más cerca de 0\n")
cat("  A medida que lambda -> inf, modelo colapsa al complemento\n")
