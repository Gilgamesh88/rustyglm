# =============================================================================
# Métodos S3 para objetos rustyglm
# =============================================================================

#' @export
print.rustyglm <- function(x, ...) {
  cat("\nModelo rustyglm\n")
  cat("---------------\n")
  cat("Familia:  ", x$family, "\n")
  cat("Fórmula:  ", deparse(x$formula), "\n")
  cat("Obs:      ", x$n, "\n")
  cat("Deviance: ", round(x$deviance, 4), "  (nula:", round(x$null.deviance, 4), ")\n")
  cat("AIC:      ", round(x$aic, 4), "\n\n")
  cat("Coeficientes:\n")
  print(round(x$coefficients, 6))
  invisible(x)
}

#' @export
coef.rustyglm <- function(object, ...) {
  object$coefficients
}

#' @export
fitted.rustyglm <- function(object, ...) {
  object$fitted.values
}

#' @export
residuals.rustyglm <- function(object,
                               type = c("response", "deviance", "pearson"),
                               ...) {
  type <- match.arg(type)
  switch(type,
         response = object$residuals,
         deviance = {
           y  <- object$y
           mu <- object$fitted.values
           sign(y - mu) * sqrt(pmax(.compute_deviance(y, mu, object$family), 0))
         },
         pearson  = {
           y  <- object$y
           mu <- object$fitted.values
           (y - mu) / sqrt(pmax(mu, 1e-10))
         }
  )
}

#' @export
summary.rustyglm <- function(object, ...) {
  coefs <- object$coefficients
  cat("\nResumen — rustyglm (", object$family, ")\n", sep = "")
  cat(rep("=", 50), "\n", sep = "")
  cat("Fórmula: ", deparse(object$formula), "\n\n")
  
  cat("Coeficientes:\n")
  df <- data.frame(
    Estimado  = round(coefs, 6),
    Exp_coef  = round(exp(coefs), 4)
  )
  print(df)
  
  cat("\n")
  cat("Deviance residual:  ", round(object$deviance, 4), "\n")
  cat("Deviance nula:      ", round(object$null.deviance, 4), "\n")
  cat("GL residual:        ", object$df.residual, "\n")
  cat("GL nulo:            ", object$df.null, "\n")
  cat("AIC:                ", round(object$aic, 4), "\n")
  
  invisible(object)
}

#' @export
predict.rustyglm <- function(object, newdata = NULL,
                             type = c("response", "link"), ...) {
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