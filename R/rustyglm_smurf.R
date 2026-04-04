# =============================================================================
# rustyglm_smurf() — SMuRF: Sparse Multi-Type Regularized Feature Modeling
# Port en Rust del algoritmo de Devriendt et al. (2021)
# Insurance: Mathematics and Economics, 96, 248-261
# =============================================================================

#' Ajusta un GLM multi-tipo regularizado usando el algoritmo SMuRF (Rust)
#'
#' Implementación en Rust puro del algoritmo FISTA proximal gradient con
#' operadores proximales específicos por tipo de predictor:
#' \itemize{
#'   \item \code{"lasso"}: soft-threshold (continuas)
#'   \item \code{"grouplasso"}: block soft-threshold (categóricas nominales)
#'   \item \code{"flasso"}: ADMM Fused Lasso (ordinales con orden natural)
#'   \item \code{"gflasso"}: ADMM Generalized Fused Lasso (ordinales, todas las diferencias)
#'   \item \code{"none"}: sin penalización
#' }
#'
#' @param formula   Fórmula R. Variables con penalización especificada en \code{pen.types}.
#' @param data      Data frame con los datos.
#' @param family    "poisson","gaussian","gamma","binomial".
#' @param offset    Vector numérico o nombre de columna para offset.
#' @param weights   Vector de pesos prior (opcional).
#' @param pen.types Named list: nombre_variable -> tipo de penalización.
#'   Variables no listadas reciben \code{"lasso"} por defecto.
#'   Ejemplo: \code{list(area = "flasso", region = "grouplasso", densidad = "lasso")}.
#' @param lambda    Parámetro global de penalización λ ≥ 0.
#' @param lambda1   Lista de pesos para la componente sparse del Fused Lasso
#'   (para Sparse Fused Lasso). NULL = sin componente sparse.
#' @param lambda2   Lista de pesos para la componente Group Lasso dentro del
#'   Fused Lasso (para Group Fused Lasso). NULL = sin componente grupo.
#' @param groups    Named list: nombre_variable -> ID de grupo (entero > 0)
#'   para Group Lasso entre predictores. NULL = sin grupos.
#' @param maxiter   Máximo de iteraciones FISTA. Default 1000.
#' @param epsilon   Tolerancia de convergencia. Default 1e-5.
#' @param step.init Paso inicial para backtracking. Default 1.0.
#' @param tau       Factor de reducción del paso (0 < tau < 1). Default 0.5.
#'
#' @return Objeto S3 de clase \code{rustyglm_smurf} con coeficientes,
#'   valores ajustados, residuales y metadatos.
#'
#' @references Devriendt, S., Antonio, K., Reynkens, T. and Verbelen, R. (2021).
#'   "Sparse Regression with Multi-type Regularized Feature Modeling",
#'   Insurance: Mathematics and Economics, 96, 248-261.
#'   \doi{10.1016/j.insmatheco.2020.11.010}
#'
#' @examples
#' \dontrun{
#' # Area ordinal (A<B<C<D<E<F) -> Fused Lasso (fusiona niveles adyacentes)
#' # VehGas nominal             -> Group Lasso (selecciona o elimina el grupo)
#' # DrivAge continuo           -> Lasso (shrinkage)
#' mod <- rustyglm_smurf(
#'   ClaimNb ~ VehAge + DrivAge + Area + VehGas,
#'   data    = datos,
#'   family  = "poisson",
#'   offset  = log(Exposure),
#'   pen.types = list(
#'     VehAge  = "flasso",
#'     DrivAge = "lasso",
#'     Area    = "flasso",
#'     VehGas  = "grouplasso"
#'   ),
#'   lambda = 0.01
#' )
#' summary(mod)
#' }
#'
#' @export
rustyglm_smurf <- function(
    formula,
    data,
    family      = "poisson",
    offset      = NULL,
    weights     = NULL,
    pen.types   = list(),
    lambda      = 0.01,
    lambda1     = NULL,
    lambda2     = NULL,
    groups      = NULL,
    maxiter     = 1000L,
    epsilon     = 1e-5,
    step.init   = 1.0,
    tau         = 0.5
) {
  # --- Validaciones ---
  if (!inherits(formula, "formula")) stop("`formula` debe ser una fórmula R.")
  if (!is.data.frame(data))         stop("`data` debe ser un data.frame.")
  family <- match.arg(tolower(family),
    c("poisson","gaussian","gamma","binomial","quasipoisson","quasibinomial"))

  # --- Parseo de fórmula ---
  mf   <- model.frame(formula, data = data, na.action = na.omit)
  y    <- as.double(model.response(mf))
  tt   <- attr(mf, "terms")
  X    <- model.matrix(tt, mf)
  n    <- nrow(X)
  p    <- ncol(X)
  vars <- attr(tt, "term.labels")  # nombres de las variables en la fórmula

  # --- Offset y weights ---
  offset_vec  <- .process_offset(offset, data, n)
  weights_vec <- if (is.null(weights)) rep(1.0, n) else as.double(weights)

  # --- Identificar bloques de columnas por predictor ---
  # Cada variable puede generar múltiples columnas (e.g. factor Area -> AreaB, AreaC, ...)
  assign_vec <- attr(X, "assign")  # mapeo columna -> término (0 = intercepto)
  col_names  <- colnames(X)

  # Mapa término -> índices de columna (0-indexed para Rust)
  term_cols <- lapply(seq_along(vars), function(k) {
    which(assign_vec == k) - 1L  # 0-indexed
  })
  names(term_cols) <- vars

  # --- Construir specs por predictor ---
  # Incluimos el intercepto como "none" (no penalizado)
  intercept_cols <- which(assign_vec == 0) - 1L  # 0-indexed

  all_specs <- list()

  # Intercepto: sin penalización
  if (length(intercept_cols) > 0) {
    all_specs[["(Intercept)"]] <- list(
      pen_type  = "none",
      col_start = min(intercept_cols),
      col_end   = max(intercept_cols) + 1L,
      pen_mat   = matrix(0, 0, 0),
      q_mat     = matrix(0, 0, 0),
      eigvals   = numeric(0),
      lambda1   = numeric(0),
      lambda2   = 0.0,
      group_id  = 0L
    )
  }

  # Variables del modelo
  for (vname in vars) {
    cols <- term_cols[[vname]]
    if (length(cols) == 0) next

    pen_type <- if (!is.null(pen.types[[vname]])) pen.types[[vname]] else "lasso"
    pen_type <- match.arg(pen_type,
      c("none","lasso","grouplasso","flasso","gflasso"))

    # Número de niveles efectivos (columnas que ocupa este predictor)
    n_par <- length(cols)

    # Penalty matrix D
    pm_list <- .build_penalty_matrix(pen_type, n_par)

    # Lambda1 por predictor (sparse component)
    l1 <- if (!is.null(lambda1[[vname]])) as.double(lambda1[[vname]]) else numeric(0)

    # Lambda2 por predictor (group component)
    l2 <- if (!is.null(lambda2[[vname]])) as.double(lambda2[[vname]])  else 0.0

    # Group ID
    gid <- if (!is.null(groups[[vname]])) as.integer(groups[[vname]]) else 0L

    all_specs[[vname]] <- list(
      pen_type  = pen_type,
      col_start = min(cols),
      col_end   = max(cols) + 1L,
      pen_mat   = pm_list$D,
      q_mat     = pm_list$Q,
      eigvals   = pm_list$eigvals,
      lambda1   = l1,
      lambda2   = l2,
      group_id  = gid
    )
  }

  # --- Aplanar specs para Rust ---
  flat <- .flatten_specs(all_specs)

  # --- Valores iniciales (warm start = glm rápido) ---
  start_coef <- tryCatch({
    glm_fam <- switch(family,
      poisson      = poisson(),
      gaussian     = gaussian(),
      gamma        = Gamma(link="log"),
      binomial     = binomial(),
      quasipoisson = quasipoisson(),
      quasibinomial = quasibinomial(),
      poisson()
    )
    m0 <- glm.fit(X, y, family = glm_fam, offset = offset_vec,
                  weights = weights_vec, control = list(maxit = 5))
    as.double(coef(m0))
  }, error = function(e) rep(0.0, p))

  # --- Llamar al solver Rust ---
  rust_out <- smurf_fit_rust(
    response      = y,
    x_flat        = as.double(X),
    nrows         = n,
    ncols         = p,
    family        = family,
    offset        = as.double(offset_vec),
    weights       = as.double(weights_vec),
    pen_types     = flat$pen_types,
    col_starts    = flat$col_starts,
    col_ends      = flat$col_ends,
    pen_mats_flat = flat$pen_mats_flat,
    pen_mat_nrows = flat$pen_mat_nrows,
    pen_mat_ncols = flat$pen_mat_ncols,
    q_mats_flat   = flat$q_mats_flat,
    q_mat_nrows   = flat$q_mat_nrows,
    q_mat_ncols   = flat$q_mat_ncols,
    eigvals_flat  = flat$eigvals_flat,
    eigval_lens   = flat$eigval_lens,
    lambda1_flat  = flat$lambda1_flat,
    lambda1_lens  = flat$lambda1_lens,
    lambda2s      = flat$lambda2s,
    group_ids     = flat$group_ids,
    lambda        = as.double(lambda),
    maxiter       = as.integer(maxiter),
    epsilon       = as.double(epsilon),
    step_init     = as.double(step.init),
    tau           = as.double(tau),
    start         = as.double(start_coef)
  )

  if (length(rust_out$coefficients) == 0)
    stop("El solver SMuRF no convergió. Prueba con lambda menor o más iteraciones.")

  coefs <- rust_out$coefficients
  names(coefs) <- col_names

  # --- Valores ajustados y residuales ---
  eta    <- as.vector(X %*% coefs) + offset_vec
  fitted <- .apply_inv_link(eta, family)
  resid  <- y - fitted

  # --- Bondad de ajuste ---
  deviance <- .compute_deviance(y, fitted, family, weights_vec)
  null_dev <- .compute_null_deviance(y, family, offset_vec, weights_vec)
  n_active <- sum(abs(coefs) > 1e-8)

  # --- Objeto S3 ---
  structure(
    list(
      coefficients   = coefs,
      fitted.values  = fitted,
      residuals      = resid,
      linear.pred    = eta,
      y              = y,
      x              = X,
      formula        = formula,
      family         = family,
      pen.types      = pen.types,
      lambda         = lambda,
      offset         = offset_vec,
      weights        = weights_vec,
      deviance       = deviance,
      null.deviance  = null_dev,
      df.residual    = n - n_active,
      df.null        = n - 1L,
      n              = n,
      n_active       = n_active,
      n_zero         = p - n_active,
      iterations     = rust_out$iterations,
      converged      = rust_out$converged,
      objective      = rust_out$objective,
      call           = match.call()
    ),
    class = "rustyglm_smurf"
  )
}

# =============================================================================
# Métodos S3
# =============================================================================

#' @export
print.rustyglm_smurf <- function(x, ...) {
  cat("\nModelo rustyglm_smurf (SMuRF — Rust)\n")
  cat(rep("-", 45), "\n", sep = "")
  cat("Familia:     ", x$family, "\n")
  cat("Fórmula:     ", deparse(x$formula), "\n")
  cat("Lambda:      ", x$lambda, "\n")
  cat("Obs:         ", x$n, "\n")
  cat("Coef activos:", x$n_active, " / eliminados:", x$n_zero, "\n")
  cat("Deviance:    ", round(x$deviance, 4),
      " (nula:", round(x$null.deviance, 4), ")\n")
  cat("Iteraciones: ", x$iterations,
      if (x$converged) " (convergió)" else " (no convergió)", "\n\n")

  cat("Tipos de penalización por predictor:\n")
  pt <- x$pen.types
  if (length(pt) > 0) {
    for (v in names(pt)) cat(sprintf("  %-20s %s\n", v, pt[[v]]))
  }

  cat("\nCoeficientes:\n")
  coef_df <- data.frame(
    Estimado  = round(x$coefficients, 6),
    Exp_coef  = round(exp(x$coefficients), 4),
    Activo    = ifelse(abs(x$coefficients) > 1e-8, "✓", "·")
  )
  print(coef_df)
  invisible(x)
}

#' @export
coef.rustyglm_smurf <- function(object, ...) object$coefficients

#' @export
fitted.rustyglm_smurf <- function(object, ...) object$fitted.values

#' @export
residuals.rustyglm_smurf <- function(object, type = c("response","pearson"), ...) {
  type <- match.arg(type)
  if (type == "response") return(object$residuals)
  (object$y - object$fitted.values) / sqrt(pmax(object$fitted.values, 1e-10))
}

#' @export
summary.rustyglm_smurf <- function(object, ...) {
  cat("\nResumen — rustyglm_smurf (", object$family, "), λ =", object$lambda, "\n")
  cat(rep("=", 55), "\n", sep = "")

  # Niveles fusionados (coef iguales dentro de un predictor)
  cat("Variables activas:", object$n_active, "/ Variables eliminadas:", object$n_zero, "\n\n")

  cat("Tipos de penalización:\n")
  for (v in names(object$pen.types))
    cat(sprintf("  %-20s %s\n", v, object$pen.types[[v]]))

  cat("\nCoeficientes:\n")
  coef_df <- data.frame(
    Estimado  = round(object$coefficients, 6),
    Exp_coef  = round(exp(object$coefficients), 4),
    Activo    = ifelse(abs(object$coefficients) > 1e-8, "si", "no")
  )
  print(coef_df)

  cat("\nDeviance residual:", round(object$deviance, 4), "\n")
  cat("Deviance nula:    ", round(object$null.deviance, 4), "\n")
  cat("Reducción (%)     ", round(100*(1 - object$deviance/object$null.deviance), 2), "%\n")
  cat("Iteraciones:      ", object$iterations, "\n")
  cat("Convergió:        ", object$converged, "\n")
  invisible(object)
}

#' @export
predict.rustyglm_smurf <- function(object, newdata = NULL,
                                    type = c("response","link"), ...) {
  type <- match.arg(type)
  if (is.null(newdata)) {
    eta <- object$linear.pred
  } else {
    mf  <- model.frame(object$formula, data = newdata, na.action = na.omit)
    X   <- model.matrix(attr(mf, "terms"), mf)
    eta <- as.vector(X %*% object$coefficients)
  }
  if (type == "link") return(eta)
  .apply_inv_link(eta, object$family)
}

# =============================================================================
# Helpers internos
# =============================================================================

#' Construye la matriz de penalización D para cada tipo
#' @keywords internal
.build_penalty_matrix <- function(pen_type, n_par) {
  if (pen_type %in% c("none","lasso","grouplasso") || n_par <= 1) {
    return(list(D = matrix(0, 0, 0), Q = matrix(0, 0, 0), eigvals = numeric(0)))
  }

  # Fused Lasso: D es la matriz de diferencias finitas (n_par-1) x n_par
  # Penaliza: sum |beta_i - beta_{i+1}|
  D <- matrix(0, nrow = n_par - 1, ncol = n_par)
  for (i in seq_len(n_par - 1)) {
    D[i, i]     <- -1
    D[i, i + 1] <-  1
  }

  if (pen_type == "gflasso") {
    # Generalized: todas las diferencias posibles (combinaciones)
    pairs <- which(lower.tri(matrix(0, n_par, n_par)), arr.ind = TRUE)
    D <- matrix(0, nrow = nrow(pairs), ncol = n_par)
    for (k in seq_len(nrow(pairs))) {
      D[k, pairs[k, 1]] <-  1
      D[k, pairs[k, 2]] <- -1
    }
  }

  # Eigendescomposición de D'D para la vía rápida del ADMM
  DtD  <- t(D) %*% D
  eig  <- tryCatch(eigen(DtD, symmetric = TRUE), error = function(e) NULL)
  if (!is.null(eig)) {
    Q      <- eig$vectors
    eigval <- pmax(eig$values, 0)  # clamp numerical negatives
  } else {
    Q      <- matrix(0, 0, 0)
    eigval <- numeric(0)
  }

  list(D = D, Q = Q, eigvals = eigval)
}

#' Aplana lista de specs a vectores paralelos para pasar a Rust
#' @keywords internal
.flatten_specs <- function(specs) {
  n_preds <- length(specs)

  pen_types  <- character(n_preds)
  col_starts <- integer(n_preds)
  col_ends   <- integer(n_preds)
  lambda2s   <- double(n_preds)
  group_ids  <- integer(n_preds)

  pen_mat_nrows <- integer(n_preds)
  pen_mat_ncols <- integer(n_preds)
  pen_mats_flat <- double(0)

  q_mat_nrows <- integer(n_preds)
  q_mat_ncols <- integer(n_preds)
  q_mats_flat <- double(0)

  eigval_lens  <- integer(n_preds)
  eigvals_flat <- double(0)

  lambda1_lens  <- integer(n_preds)
  lambda1_flat  <- double(0)

  for (j in seq_along(specs)) {
    sp <- specs[[j]]
    pen_types[j]  <- sp$pen_type
    col_starts[j] <- as.integer(sp$col_start)
    col_ends[j]   <- as.integer(sp$col_end)
    lambda2s[j]   <- sp$lambda2
    group_ids[j]  <- sp$group_id

    D <- sp$pen_mat
    if (is.matrix(D) && nrow(D) > 0 && ncol(D) > 0) {
      pen_mat_nrows[j] <- nrow(D)
      pen_mat_ncols[j] <- ncol(D)
      pen_mats_flat    <- c(pen_mats_flat, as.double(D))  # R stores col-major
    }

    Q <- sp$q_mat
    if (is.matrix(Q) && nrow(Q) > 0 && ncol(Q) > 0) {
      q_mat_nrows[j] <- nrow(Q)
      q_mat_ncols[j] <- ncol(Q)
      q_mats_flat    <- c(q_mats_flat, as.double(Q))
    }

    ev <- sp$eigvals
    if (length(ev) > 0) {
      eigval_lens[j] <- length(ev)
      eigvals_flat   <- c(eigvals_flat, ev)
    }

    l1 <- sp$lambda1
    if (length(l1) > 0) {
      lambda1_lens[j] <- length(l1)
      lambda1_flat    <- c(lambda1_flat, l1)
    }
  }

  list(
    pen_types     = pen_types,
    col_starts    = col_starts,
    col_ends      = col_ends,
    pen_mats_flat = pen_mats_flat,
    pen_mat_nrows = pen_mat_nrows,
    pen_mat_ncols = pen_mat_ncols,
    q_mats_flat   = q_mats_flat,
    q_mat_nrows   = q_mat_nrows,
    q_mat_ncols   = q_mat_ncols,
    eigvals_flat  = eigvals_flat,
    eigval_lens   = eigval_lens,
    lambda1_flat  = lambda1_flat,
    lambda1_lens  = lambda1_lens,
    lambda2s      = lambda2s,
    group_ids     = group_ids
  )
}
