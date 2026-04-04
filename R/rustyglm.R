#' Ajusta un GLM usando el motor RustyStats (Rust)
#'
#' @param formula      Fórmula R estándar.
#' @param data         Data frame con los datos.
#' @param family       "poisson","gaussian","gamma","binomial","negbinomial".
#' @param offset       Vector numérico o nombre de columna para offset.
#' @param weights      Vector numérico de pesos prior (opcional).
#' @param regularization Tipo: "none","ridge","lasso","elasticnet".
#' @param lambda       Fuerza de regularización. Default 0.01.
#' @param l1_ratio     Ratio L1 para ElasticNet (1.0=Lasso, 0.0=Ridge).
#' @export
rustyglm <- function(formula, data, family = "poisson",
                     offset = NULL, weights = NULL,
                     regularization = "none", lambda = 0.01, l1_ratio = 0.5) {
  
  if (!inherits(formula, "formula")) stop("`formula` debe ser una fórmula R.")
  if (!is.data.frame(data))         stop("`data` debe ser un data.frame.")
  
  family <- match.arg(tolower(family),
                      c("poisson","gaussian","gamma","binomial",
                        "negbinomial","quasipoisson","quasibinomial"))
  
  regularization <- match.arg(tolower(regularization),
                              c("none","ridge","lasso","elasticnet"))
  
  mf <- model.frame(formula, data = data, na.action = na.omit)
  y  <- model.response(mf)
  X  <- model.matrix(attr(mf, "terms"), mf)
  n  <- nrow(X)
  p  <- ncol(X)
  
  offset_vec  <- .process_offset(offset, data, n)
  weights_vec <- if (is.null(weights)) rep(1.0, n) else as.double(weights)
  
  if (length(weights_vec) != n)
    stop("La longitud de `weights` no coincide con los datos.")
  
  coefs <- .call_rust_glm(y, X, family, offset_vec, regularization, lambda, l1_ratio)
  
  if (length(coefs) == 0)
    stop("El solver Rust no pudo ajustar el modelo.")
  
  names(coefs) <- colnames(X)
  
  eta    <- as.vector(X %*% coefs) + offset_vec
  fitted <- .apply_inv_link(eta, family)
  resid  <- as.double(y) - fitted
  
  deviance <- .compute_deviance(as.double(y), fitted, family, weights_vec)
  null_dev <- .compute_null_deviance(as.double(y), family, offset_vec, weights_vec)
  aic      <- deviance + 2 * p
  
  structure(
    list(
      coefficients   = coefs,
      fitted.values  = fitted,
      residuals      = resid,
      linear.pred    = eta,
      y              = as.double(y),
      x              = X,
      formula        = formula,
      family         = family,
      regularization = regularization,
      lambda         = lambda,
      l1_ratio       = l1_ratio,
      offset         = offset_vec,
      weights        = weights_vec,
      deviance       = deviance,
      null.deviance  = null_dev,
      aic            = aic,
      df.residual    = n - p,
      df.null        = n - 1L,
      n              = n,
      call           = match.call()
    ),
    class = "rustyglm"
  )
}

#' @keywords internal
.process_offset <- function(offset, data, n) {
  if (is.null(offset)) return(rep(0.0, n))
  if (is.character(offset) && length(offset) == 1L) {
    if (!offset %in% names(data))
      stop("La columna de offset '", offset, "' no existe en `data`.")
    return(as.double(data[[offset]]))
  }
  if (is.numeric(offset)) {
    if (length(offset) != n)
      stop("Longitud del offset (", length(offset), ") != datos (", n, ").")
    return(as.double(offset))
  }
  stop("`offset` debe ser vector numérico o nombre de columna.")
}

#' @keywords internal
.call_rust_glm <- function(y, X, family, offset_vec,
                           regularization = "none", lambda = 0.01, l1_ratio = 0.5) {
  glm_fit_rust(
    response = as.double(y),
    x_flat   = as.double(X),
    nrows    = nrow(X),
    ncols    = ncol(X),
    family   = family,
    offset   = as.double(offset_vec),
    reg_type = regularization,
    lambda   = as.double(lambda),
    l1_ratio = as.double(l1_ratio)
  )
}

#' @keywords internal
.apply_inv_link <- function(eta, family) {
  switch(family,
         poisson = , quasipoisson = , gamma = , negbinomial = exp(eta),
         binomial = , quasibinomial = 1 / (1 + exp(-eta)),
         gaussian = eta,
         eta
  )
}

#' @keywords internal
.compute_deviance <- function(y, mu, family, weights = NULL) {
  w <- if (is.null(weights)) rep(1.0, length(y)) else weights
  unit_dev <- switch(family,
                     poisson = , quasipoisson  = 2 * (ifelse(y == 0, 0, y * log(y / mu)) - (y - mu)),
                     gaussian    = (y - mu)^2,
                     gamma       = 2 * (-log(y / mu) + (y - mu) / mu),
                     binomial = , quasibinomial = -2 * (y * log(mu) + (1 - y) * log(1 - mu)),
                     negbinomial = 2 * (ifelse(y == 0, 0, y * log(y / mu)) - (y - mu)),
                     (y - mu)^2
  )
  sum(w * unit_dev)
}

#' @keywords internal
.compute_null_deviance <- function(y, family, offset, weights) {
  mu_null <- rep(mean(y), length(y))
  .compute_deviance(y, mu_null, family, weights)
}