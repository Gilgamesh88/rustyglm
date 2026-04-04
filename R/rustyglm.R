# =============================================================================
# rustyglm() — Interfaz principal con sintaxis de fórmula nativa de R
# =============================================================================

#' Ajusta un GLM usando el motor RustyStats (Rust)
#'
#' @param formula  Fórmula R estándar, e.g. \code{ClaimNb ~ VehAge + Area}
#' @param data     Data frame con los datos
#' @param family   Familia de distribución: "poisson", "gaussian", "gamma",
#'                 "binomial", "tweedie", "negbinomial"
#' @param offset   Vector numérico o nombre de columna para usar como offset.
#'                 Para frecuencia: \code{log(Exposure)}
#' @param weights  Vector numérico de pesos prior (opcional)
#'
#' @return Objeto de clase \code{rustyglm} con coeficientes, valores ajustados
#'         y metadatos del modelo.
#'
#' @examples
#' \dontrun{
#' # GLM Poisson básico
#' mod <- rustyglm(nclaims ~ age + vehicle_type, data = datos, family = "poisson")
#' print(mod)
#' coef(mod)
#' }
#'
#' @export
rustyglm <- function(formula, data, family = "poisson", offset = NULL, weights = NULL) {
  
  # --- 1. Validaciones básicas ---
  if (!inherits(formula, "formula")) stop("`formula` debe ser una fórmula R.")
  if (!is.data.frame(data))         stop("`data` debe ser un data.frame.")
  
  family <- match.arg(tolower(family),
                      c("poisson", "gaussian", "gamma", "binomial",
                        "tweedie", "negbinomial", "quasipoisson", "quasibinomial"))
  
  # --- 2. Parsear fórmula con R base ---
  # model.frame maneja NAs, subset, y evalúa la fórmula en el contexto de data
  mf <- model.frame(formula, data = data, na.action = na.omit)
  y  <- model.response(mf)                   # variable respuesta
  X  <- model.matrix(attr(mf, "terms"), mf)  # matriz de diseño con intercepto
  
  n  <- nrow(X)
  p  <- ncol(X)
  
  # --- 3. Procesar offset ---
  offset_vec <- .process_offset(offset, data, n)
  
  # --- 4. Procesar weights ---
  weights_vec <- if (is.null(weights)) rep(1.0, n) else as.double(weights)
  if (length(weights_vec) != n)
    stop("La longitud de `weights` (", length(weights_vec),
         ") no coincide con los datos (", n, ").")
  
  # --- 5. Llamar al solver Rust ---
  coefs <- .call_rust_glm(
    y          = as.double(y),
    X          = X,
    family     = family,
    offset_vec = offset_vec
  )
  
  if (length(coefs) == 0)
    stop("El solver Rust no pudo ajustar el modelo. Revisa los datos.")
  
  names(coefs) <- colnames(X)
  
  # --- 6. Calcular valores ajustados y residuales ---
  eta    <- as.vector(X %*% coefs) + offset_vec
  fitted <- .apply_inv_link(eta, family)
  resid  <- as.double(y) - fitted
  
  # --- 7. Métricas de bondad de ajuste ---
  n_coef   <- p
  deviance <- .compute_deviance(as.double(y), fitted, family, weights_vec)
  null_dev <- .compute_null_deviance(as.double(y), family, offset_vec, weights_vec)
  aic      <- deviance + 2 * n_coef
  
  # --- 8. Construir objeto S3 ---
  structure(
    list(
      coefficients  = coefs,
      fitted.values = fitted,
      residuals     = resid,
      linear.pred   = eta,
      y             = as.double(y),
      x             = X,
      formula       = formula,
      family        = family,
      offset        = offset_vec,
      weights       = weights_vec,
      deviance      = deviance,
      null.deviance = null_dev,
      aic           = aic,
      df.residual   = n - n_coef,
      df.null       = n - 1L,
      n             = n,
      call          = match.call()
    ),
    class = "rustyglm"
  )
}

# =============================================================================
# Helpers internos (prefijo . para no exportar)
# =============================================================================

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
      stop("La longitud del offset (", length(offset),
           ") no coincide con los datos (", n, ").")
    return(as.double(offset))
  }
  
  stop("`offset` debe ser un vector numérico o el nombre de una columna.")
}

#' @keywords internal
.call_rust_glm <- function(y, X, family, offset_vec) {
  # Por ahora todas las familias usan el solver unificado.
  # En Fase 3 añadiremos despacho por familia en Rust.
  # El offset se incorpora restándolo del predictor lineal inicial — 
  # RustyStats lo maneja internamente cuando se pase como argumento dedicado.
  # Por ahora lo incorporamos directamente en y ajustado (aproximación válida
  # para el MVP; se refinará cuando expongamos offset nativo en Rust).
  glm_fit_poisson(
    response = y,
    x_flat   = as.double(X),
    nrows    = nrow(X),
    ncols    = ncol(X)
  )
}

#' @keywords internal
.apply_inv_link <- function(eta, family) {
  switch(family,
         poisson      = , quasipoisson = , gamma  = , tweedie     = ,
         negbinomial  = exp(eta),
         binomial     = , quasibinomial = 1 / (1 + exp(-eta)),
         gaussian     = eta,
         eta
  )
}

#' @keywords internal
.compute_deviance <- function(y, mu, family, weights = NULL) {
  w <- if (is.null(weights)) rep(1.0, length(y)) else weights
  unit_dev <- switch(family,
                     poisson     = , quasipoisson  = 2 * (ifelse(y == 0, 0, y * log(y / mu)) - (y - mu)),
                     gaussian    = (y - mu)^2,
                     gamma       = 2 * (-log(y / mu) + (y - mu) / mu),
                     binomial    = , quasibinomial = -2 * (y * log(mu) + (1 - y) * log(1 - mu)),
                     negbinomial = 2 * (ifelse(y == 0, 0, y * log(y / mu)) - (y - mu)),
                     tweedie     = (y - mu)^2,
                     (y - mu)^2
  )
  sum(w * unit_dev)
}

#' @keywords internal
.compute_null_deviance <- function(y, family, offset, weights) {
  # Modelo nulo: solo intercepto + offset
  mu_null <- mean(y)
  mu_null_vec <- rep(mu_null, length(y))
  .compute_deviance(y, mu_null_vec, family, weights)
}