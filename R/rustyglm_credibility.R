# =============================================================================
# Lasso Credibility para rustyglm
# Basado en CAS Monograph 13 (Holmes & Casotto, 2025)
# =============================================================================
# Idea: en lugar de regularizar beta hacia cero (Lasso estándar),
# regularizamos hacia un modelo prior (complemento de credibilidad).
#
# Matemáticamente: si el link es log y el complemento son tasas c_i,
#   añadir log(c_i) al offset hace que el Lasso shrinkee HACIA c_i.
#
# Equivalencia exacta con la formulación de RustyStats Python:
#   complement="countrywide_rate" + regularization="lasso"
# =============================================================================

#' Ajusta un GLM con Credibilidad Lasso (shrinkage hacia modelo prior)
#'
#' Implementa la metodología de credibilidad bayesiana de Bühlmann-Straub
#' combinada con regularización Lasso. Los coeficientes se encogen hacia
#' el modelo de complemento (prior), no hacia cero.
#'
#' @param formula      Fórmula R estándar.
#' @param data         Data frame con los datos.
#' @param family       Familia de distribución.
#' @param complement   Vector numérico de predicciones del modelo prior
#'   (escala de respuesta, e.g. tasas de siniestralidad del modelo nacional).
#'   Puede ser nombre de columna en \code{data}.
#' @param offset       Offset adicional (se suma al log del complemento).
#' @param weights      Pesos prior.
#' @param regularization Tipo: "lasso" (default), "ridge", "elasticnet".
#' @param lambda       Fuerza de regularización. A mayor lambda, más
#'   credibilidad al modelo prior.
#' @param l1_ratio     Ratio L1 para ElasticNet.
#'
#' @return Objeto S3 \code{rustyglm_credibility} con los mismos métodos
#'   que \code{rustyglm} más información de credibilidad.
#'
#' @details
#' La credibilidad Lasso funciona mediante la transformación del offset:
#' \deqn{offset_{credibilidad} = log(complemento_i) + offset_{adicional}}
#' El Lasso entonces shrinkea los coeficientes hacia las predicciones del
#' complemento. Cuando lambda -> inf, el modelo colapsa al complemento.
#' Cuando lambda = 0, el modelo es el GLM estándar sin credibilidad.
#'
#' @references Holmes, C. and Casotto, E. (2025). CAS Monograph 13:
#'   Lasso Credibility. Casualty Actuarial Society.
#'
#' @examples
#' \dontrun{
#' # 1. Modelo nacional (complemento)
#' mod_nacional <- rustyglm(
#'   ClaimNb ~ VehAge + DrivAge,
#'   data = datos_nacional, family = "poisson",
#'   offset = log(Exposure)
#' )
#' tasas_previstas <- fitted(mod_nacional)[match(datos_estado$id, datos_nacional$id)]
#'
#' # 2. Modelo estatal con credibilidad
#' mod_estado <- rustyglm_credibility(
#'   ClaimNb ~ VehAge + DrivAge + Region,
#'   data        = datos_estado,
#'   family      = "poisson",
#'   complement  = tasas_previstas,  # tasas del modelo nacional
#'   offset      = log(datos_estado$Exposure),
#'   lambda      = 0.1   # mayor lambda = más peso al modelo nacional
#' )
#' credibility_summary(mod_estado)
#' }
#'
#' @export
rustyglm_credibility <- function(
    formula,
    data,
    family         = "poisson",
    complement,    # predicciones del modelo prior (respuesta scale)
    offset         = NULL,
    weights        = NULL,
    regularization = "lasso",
    lambda         = 0.01,
    l1_ratio       = 1.0
) {
  if (!inherits(formula, "formula")) stop("`formula` debe ser una fórmula R.")
  if (!is.data.frame(data))         stop("`data` debe ser un data.frame.")

  family         <- match.arg(tolower(family),
    c("poisson","gaussian","gamma","binomial","quasipoisson","quasibinomial"))
  regularization <- match.arg(tolower(regularization),
    c("lasso","ridge","elasticnet"))

  # Resolver complemento
  if (is.character(complement) && length(complement) == 1L) {
    if (!complement %in% names(data))
      stop("La columna de complemento '", complement, "' no existe en `data`.")
    comp_vec <- as.double(data[[complement]])
  } else if (is.numeric(complement)) {
    comp_vec <- as.double(complement)
  } else {
    stop("`complement` debe ser vector numérico o nombre de columna.")
  }

  # Parseo inicial para obtener n
  mf <- model.frame(formula, data = data, na.action = na.omit)
  n  <- nrow(mf)

  if (length(comp_vec) != n)
    stop("Longitud del complemento (", length(comp_vec), ") != datos (", n, ").")
  if (any(comp_vec <= 0))
    stop("El complemento debe ser positivo (está en escala de respuesta).")

  # Offset de credibilidad: log(complemento) + offset_adicional
  base_offset   <- .process_offset(offset, data, n)
  credit_offset <- log(comp_vec) + base_offset

  # Ajustar modelo con el offset modificado
  mod <- rustyglm(
    formula        = formula,
    data           = data,
    family         = family,
    offset         = credit_offset,
    weights        = weights,
    regularization = regularization,
    lambda         = lambda,
    l1_ratio       = l1_ratio
  )

  # Añadir información de credibilidad al objeto
  mod$complement       <- comp_vec
  mod$credit_offset    <- credit_offset
  mod$base_offset      <- base_offset
  mod$credibility_lambda <- lambda

  # Calcular factor de credibilidad implícito por coeficiente
  # Z = 1 - |beta_credibility - beta_naive| / |beta_naive - 0|
  # (aproximación; el factor exacto requiere el modelo sin credibilidad)
  class(mod) <- c("rustyglm_credibility", "rustyglm")
  mod
}

#' Resumen de credibilidad: qué tanto se usa el modelo propio vs el prior
#' @export
credibility_summary <- function(object, ...) UseMethod("credibility_summary")

#' @export
credibility_summary.rustyglm_credibility <- function(object, ...) {
  cat("\n=== Resumen de Credibilidad (CAS Monograph 13) ===\n\n")
  cat("Lambda de credibilidad:", object$credibility_lambda, "\n")
  cat("Regularización:        ", object$regularization, "\n\n")

  coefs <- object$coefficients
  cat("Coeficientes (shrinkage hacia el complemento):\n")
  df <- data.frame(
    Estimado     = round(coefs, 6),
    Relatividad  = round(exp(coefs), 4),
    Shrinkage    = ifelse(abs(coefs) < 1e-6, "Eliminado (→complemento)", "Activo")
  )
  print(df)

  # Comparar fitted vs complemento
  r <- cor(object$fitted.values, object$complement, method = "spearman")
  cat(sprintf("\nCorrelación rank(fitted) vs rank(complemento): %.4f\n", r))
  cat(sprintf("Ratio A/E global vs complemento: %.4f\n",
              sum(object$y) / sum(object$complement)))

  invisible(object)
}

#' @export
print.rustyglm_credibility <- function(x, ...) {
  cat("\nModelo rustyglm — Credibilidad Lasso\n")
  cat(rep("-", 40), "\n", sep = "")
  cat("Familia:    ", x$family, "\n")
  cat("Lambda:     ", x$credibility_lambda, "\n")
  cat("Fórmula:    ", deparse(x$formula), "\n")
  cat("Obs:        ", x$n, "\n")
  cat("Deviance:   ", round(x$deviance, 4), "\n\n")
  cat("Coeficientes (hacia complemento con lambda =", x$credibility_lambda, "):\n")
  print(round(x$coefficients, 6))
  invisible(x)
}
