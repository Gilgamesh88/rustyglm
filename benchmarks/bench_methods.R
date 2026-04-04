# =============================================================================
# Benchmark: IRLS vs Ridge vs Lasso vs ElasticNet
# Dataset: freMTPL2freq — 678k observaciones
# Pregunta: ¿cuánto cuesta la regularización vs IRLS puro?
# =============================================================================

library(rustyglm)
library(CASdatasets)
library(bench)
library(glmnet)  # install.packages("glmnet") si no está

data(freMTPL2freq)

dat <- freMTPL2freq
dat$Exposure <- as.double(dat$Exposure)
dat$Area     <- as.factor(dat$Area)
dat$VehGas   <- as.factor(dat$VehGas)

formula_freq <- ClaimNb ~ VehAge + DrivAge + BonusMalus + VehPower + Area + VehGas + Density
off          <- log(dat$Exposure)

# Preparar X e y para glmnet (necesita matrices explícitas)
mf  <- model.frame(formula_freq, data = dat)
y   <- as.double(model.response(mf))
X   <- model.matrix(attr(mf, "terms"), mf)

cat("=== Benchmark métodos de ajuste — freMTPL2freq (n=678k) ===\n\n")

# Lambda fija para comparación justa (misma penalización en todos)
LAMBDA <- 0.001

bm <- bench::mark(
  # IRLS puro (sin regularización)
  IRLS = rustyglm(formula_freq, data = dat, family = "poisson",
                  offset = off, regularization = "none"),
  
  # Ridge (IRLS modificado con penalización L2)
  Ridge = rustyglm(formula_freq, data = dat, family = "poisson",
                   offset = off, regularization = "ridge", lambda = LAMBDA),
  
  # Lasso (coordinate descent, selección de variables)
  Lasso = rustyglm(formula_freq, data = dat, family = "poisson",
                   offset = off, regularization = "lasso", lambda = LAMBDA),
  
  # ElasticNet (coordinate descent, L1+L2)
  ElasticNet = rustyglm(formula_freq, data = dat, family = "poisson",
                        offset = off, regularization = "elasticnet",
                        lambda = LAMBDA, l1_ratio = 0.5),
  
  # glmnet como referencia externa (también coordinate descent)
  glmnet_lasso = glmnet::glmnet(X, y, family = "poisson",
                                offset = off, alpha = 1,
                                lambda = LAMBDA, standardize = FALSE),
  
  glmnet_ridge = glmnet::glmnet(X, y, family = "poisson",
                                offset = off, alpha = 0,
                                lambda = LAMBDA, standardize = FALSE),
  
  iterations = 5,
  check = FALSE
)

print(bm[, c("expression", "min", "median", "mem_alloc", "n_itr")])

# =============================================================================
# Comparación numérica de coeficientes entre métodos
# =============================================================================
cat("\n=== Coeficientes por método (λ =", LAMBDA, ") ===\n")

mod_irls <- rustyglm(formula_freq, data = dat, family = "poisson",
                     offset = off, regularization = "none")
mod_ridge <- rustyglm(formula_freq, data = dat, family = "poisson",
                      offset = off, regularization = "ridge", lambda = LAMBDA)
mod_lasso <- rustyglm(formula_freq, data = dat, family = "poisson",
                      offset = off, regularization = "lasso", lambda = LAMBDA)
mod_enet  <- rustyglm(formula_freq, data = dat, family = "poisson",
                      offset = off, regularization = "elasticnet",
                      lambda = LAMBDA, l1_ratio = 0.5)
mod_r     <- glm(formula_freq, data = dat, family = poisson(), offset = log(Exposure))

comparison <- data.frame(
  glm_R    = round(coef(mod_r),     6),
  IRLS     = round(coef(mod_irls),  6),
  Ridge    = round(coef(mod_ridge), 6),
  Lasso    = round(coef(mod_lasso), 6),
  ElasticNet = round(coef(mod_enet), 6)
)
print(comparison)

# Coeficientes puestos a cero por Lasso
zeros_lasso <- sum(coef(mod_lasso) == 0)
zeros_enet  <- sum(coef(mod_enet)  == 0)
cat(sprintf("\nVariables eliminadas por Lasso:      %d de %d\n", zeros_lasso, length(coef(mod_lasso))))
cat(sprintf("Variables eliminadas por ElasticNet: %d de %d\n", zeros_enet,  length(coef(mod_enet))))

# =============================================================================
# Efecto del lambda en Lasso (shrinkage path)
# =============================================================================
cat("\n=== Path de Lasso: coeficientes vs lambda ===\n")

lambdas <- c(0.0001, 0.001, 0.01, 0.05, 0.1, 0.5)
lasso_path <- sapply(lambdas, function(lam) {
  m <- rustyglm(formula_freq, data = dat, family = "poisson",
                offset = off, regularization = "lasso", lambda = lam)
  coef(m)
})
colnames(lasso_path) <- paste0("λ=", lambdas)
print(round(lasso_path, 5))

cat("\nVariables activas (≠0) por lambda:\n")
activas <- apply(lasso_path, 2, function(x) sum(x != 0))
print(activas)