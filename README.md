# rustyglm <img src="man/figures/logo.png" align="right" height="139" alt="" />

> R bindings for [RustyStats](https://github.com/PricingFrontier/rustystats) — high-performance GLM with native R formula syntax

[![Rust](https://img.shields.io/badge/rust-1.94-orange?logo=rust)](https://www.rust-lang.org/)
[![R](https://img.shields.io/badge/R-4.5-blue?logo=r)](https://www.r-project.org/)
[![extendr](https://img.shields.io/badge/extendr-0.8.1-green)](https://extendr.github.io/)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)

## Overview

`rustyglm` exposes the [RustyStats](https://github.com/PricingFrontier/rustystats) Rust GLM engine to R using native R formula syntax via [extendr](https://extendr.github.io/). It provides significant memory efficiency gains over base R's `glm()` while producing numerically identical results.

```r
# Native R formula syntax — just like glm()
mod <- rustyglm(ClaimNb ~ VehAge + DrivAge + BonusMalus + Area,
                data   = freMTPL2freq,
                family = "poisson",
                offset = log(Exposure))

# Or with SMuRF: per-predictor penalty types
mod <- rustyglm_smurf(
  ClaimNb ~ VehAge_f + DrivAge + Area_f + VehGas_f,
  data      = datos,
  family    = "poisson",
  offset    = log(Exposure),
  pen.types = list(
    VehAge_f = "flasso",    # ordinal → fused lasso (merges adjacent levels)
    DrivAge  = "lasso",     # continuous → shrinkage
    Area_f   = "flasso",    # ordinal → fused lasso
    VehGas_f = "grouplasso" # nominal → group lasso (select or eliminate)
  ),
  lambda = 0.01
)
```

## Architecture

```
R (formula, model.matrix) → extendr → RustyStats core (Rust IRLS/FISTA) → R
```

| Layer | Technology | Role |
|---|---|---|
| Math engine | **RustyStats** (Rust, AGPL-3.0) | IRLS, FISTA, ADMM solvers |
| Binding | **extendr** / **rextendr** 0.8.1 | Rust ↔ R bridge |
| Interface | **R** (formulas, S3, tidyverse) | User-facing API |

The R formula parsing is done entirely in R using `model.matrix()` and `model.frame()`. No formula parsing happens in Rust.

## Benchmarks

Dataset: `freMTPL2freq` (French Motor Third Party Liability, ~678k policies)

### Speed & Memory — Single model fit

| Method | n=10k | n=50k | n=100k | n=678k | Memory (678k) |
|---|---|---|---|---|---|
| rustyglm | 95ms | 270ms | 470ms | 2.98s | **478 MB** |
| glm() base R | 34ms | 206ms | 373ms | 2.94s | 3.74 GB |

**Memory reduction: 7.8× less RAM than base R** at full scale.

### SMuRF: Sparse Multi-Type Regularization

| Method | n=20k median | Memory | vs smurf R |
|---|---|---|---|
| **SMuRF Rust** (rustyglm) | 902ms | **166 MB** | **6.1× faster** |
| smurf R (CRAN) | 5.5s | 3.17 GB | baseline |
| rustyglm Lasso | 238ms | 15 MB | — |
| glm() base R | 78ms | 113 MB | — |

### Numerical accuracy

Difference vs `glm()` base R: **< 1e-6** (IRLS tolerance).

```r
max(abs(coef(mod_rust) - coef(mod_r)))
#> [1] 9.52e-06
```

## Installation

### Prerequisites

- R ≥ 4.1
- Rtools45 (Windows) or build tools (macOS/Linux)
- Rust via [rustup](https://rustup.rs/)

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Windows: add GNU target (required for Rtools)
rustup target add x86_64-pc-windows-gnu
```

### From GitHub

```r
# Install dependencies
install.packages(c("rextendr", "devtools"))

# Install rustyglm
devtools::install_github("Gilgamesh88/rustyglm")
```

### Development install

```r
git clone https://github.com/Gilgamesh88/rustyglm.git
cd rustyglm

# In R:
rextendr::document()   # compile Rust + generate wrappers
devtools::install()
```

## Features

### 1. Standard GLM with regularization

```r
library(rustyglm)

# No regularization (equivalent to glm())
mod <- rustyglm(ClaimNb ~ VehAge + DrivAge + BonusMalus + Area,
                data   = datos,
                family = "poisson",
                offset = log(Exposure))

# Ridge
mod_ridge <- rustyglm(..., regularization = "ridge",      lambda = 0.01)

# Lasso
mod_lasso <- rustyglm(..., regularization = "lasso",      lambda = 0.01)

# Elastic Net
mod_enet  <- rustyglm(..., regularization = "elasticnet", lambda = 0.01, l1_ratio = 0.5)

# S3 methods
print(mod)
summary(mod)
coef(mod)
fitted(mod)
residuals(mod, type = "pearson")
predict(mod, newdata = nuevos_datos, type = "response")
```

**Supported families:** `"poisson"`, `"gaussian"`, `"gamma"`, `"binomial"`, `"negbinomial"`, `"quasipoisson"`, `"quasibinomial"`

### 2. SMuRF — Sparse Multi-Type Regularized Feature Modeling

Implementation of Devriendt et al. (2021) using Rust FISTA proximal gradient with per-predictor penalty operators.

```r
dat$VehAge_f <- factor(dat$VehAge)   # ordinal → Fused Lasso
dat$Area_f   <- factor(dat$Area)     # ordinal → Fused Lasso
dat$VehGas_f <- factor(dat$VehGas)   # nominal → Group Lasso

mod_smurf <- rustyglm_smurf(
  formula   = ClaimNb ~ VehAge_f + DrivAge + Area_f + VehGas_f + Density,
  data      = datos,
  family    = "poisson",
  offset    = log(Exposure),
  pen.types = list(
    VehAge_f = "flasso",    # fuse adjacent ordinal levels
    DrivAge  = "lasso",     # standard L1 shrinkage
    Area_f   = "flasso",    # fuse adjacent geographic zones
    VehGas_f = "grouplasso",# select or eliminate entire factor
    Density  = "lasso"
  ),
  lambda  = 0.01,
  maxiter = 2000L
)

summary(mod_smurf)
# Shows: active coefficients, fused levels, deviance reduction
```

**Penalty types:**

| Type | Predictor kind | Effect |
|---|---|---|
| `"lasso"` | Continuous | L1 shrinkage toward zero |
| `"grouplasso"` | Nominal factor | Eliminate or keep entire group |
| `"flasso"` | Ordinal factor | Fuse adjacent levels automatically |
| `"gflasso"` | Ordinal factor | Fuse all pairs of levels |
| `"none"` | Any | No penalization |

**Algorithm:** FISTA (Beck & Teboulle 2009) with Nesterov acceleration and backtracking line search. Proximal operators: soft-threshold (Lasso), block soft-threshold (Group Lasso), ADMM with adaptive ρ (Zhu 2017) for Fused Lasso.

### 3. Lasso Credibility

Implementation of the Lasso Credibility methodology (Holmes & Casotto, CAS Monograph 13, 2025): shrink coefficients toward a prior model instead of toward zero.

```r
# Step 1: fit national (prior) model
mod_nacional <- rustyglm(
  ClaimNb ~ VehAge + DrivAge + BonusMalus + Area,
  data   = datos_nacional,
  family = "poisson",
  offset = log(Exposure)
)

# Step 2: predict national rates for the local portfolio
prior_rates <- predict(mod_nacional, newdata = datos_estado, type = "response")

# Step 3: fit local model with credibility shrinkage toward national rates
mod_estado <- rustyglm_credibility(
  ClaimNb ~ VehAge + DrivAge + BonusMalus + Area + Region,
  data       = datos_estado,
  family     = "poisson",
  complement = prior_rates,  # predictions from national model
  offset     = log(Exposure),
  lambda     = 0.1           # higher = more weight on national model
)

# Inspect credibility results
credibility_summary(mod_estado)
# Shows: coefficients, relativities, A/E vs complement
```

**How it works:** Adding `log(complement)` to the offset transforms the Lasso shrinkage problem so that coefficients shrink toward the prior predictions instead of toward zero. When `lambda → ∞`, the local model collapses to the national complement.

## Roadmap

### Completed ✓
- **Phase 0** — Rust scaffold: extendr binding, end-to-end compilation
- **Phase 1** — R formula interface: `rustyglm()` with S3 methods
- **Phase 2** — Benchmarks: speed/memory vs glm(), glmnet, smurf
- **Phase 3 (partial)** — Regularization: Ridge, Lasso, ElasticNet, SMuRF, Credibility

### In progress
- **Phase 3** — Extended syntax: `bs()`, `ns()` with monotonicity via `SmoothGLMConfig`
- **Phase 4** — Diagnostics: `tidy()`, `glance()`, `augment()` via broom; AvE plots

### Planned
- **Phase 5** — Actuarial extensions:
  - `faer` for linear algebra in ADMM (faster Fused Lasso)
  - SMuRF lambda selection via cross-validation
  - `rustyglm_to_pmml()`, `rustyglm_to_onnx()` export
  - INLA in Rust (future project)

## Dependencies

### Rust crates

| Crate | Version | Role |
|---|---|---|
| `rustystats-core` | 0.6.3 | GLM math engine (IRLS, coord. descent) |
| `extendr-api` | 0.8.1 | Rust ↔ R bridge |
| `ndarray` | 0.17 | N-dimensional arrays |
| `nalgebra` | 0.34 | Linear algebra (ADMM matrix inverse) |
| `rayon` | 1.11 | Parallel iterators |
| `statrs` | 0.18 | Statistical distributions |
| `thiserror` | 2.0 | Error handling |

### R packages

- `rextendr` — development only (compile Rust)
- `bench` — benchmarks
- `CASdatasets` — `freMTPL2freq` dataset
- `smurf` — comparison benchmarks

## References

- Devriendt, S., Antonio, K., Reynkens, T. and Verbelen, R. (2021). "Sparse Regression with Multi-type Regularized Feature Modeling." *Insurance: Mathematics and Economics*, 96, 248–261. [doi:10.1016/j.insmatheco.2020.11.010](https://doi.org/10.1016/j.insmatheco.2020.11.010)

- Holmes, C. and Casotto, E. (2025). *CAS Monograph 13: Lasso Credibility*. Casualty Actuarial Society.

- Beck, A. and Teboulle, M. (2009). "A fast iterative shrinkage-thresholding algorithm for linear inverse problems." *SIAM Journal on Imaging Sciences*, 2(1), 183–202.

- Zhu, Y. (2017). "An augmented ADMM algorithm with application to the generalized lasso problem." *Journal of Computational and Graphical Statistics*, 26(1), 195–204.

- RustyStats: [github.com/PricingFrontier/rustystats](https://github.com/PricingFrontier/rustystats)

- extendr: [extendr.github.io](https://extendr.github.io/)

## License

AGPL-3.0 — consistent with RustyStats upstream license.

---

*Built by an actuary learning Rust. Contributions welcome.*
