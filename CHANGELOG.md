# Changelog

All notable changes to `rustyglm` are documented here.

## [0.3.0] — 2026-04-03

### Added
- **SMuRF algorithm** (`rustyglm_smurf()`): Sparse Multi-Type Regularized Feature Modeling
  - Pure Rust implementation of FISTA proximal gradient (Devriendt et al. 2021)
  - Proximal operators: Lasso (soft-threshold), Group Lasso (block soft-threshold),
    Fused Lasso and Generalized Fused Lasso (ADMM with adaptive ρ, Zhu 2017)
  - Per-predictor penalty specification via `pen.types` named list
  - Native R formula interface with `model.matrix()` parsing
  - S3 methods: `print`, `summary`, `coef`, `fitted`, `residuals`, `predict`
  - **10.5× faster** than R `smurf` package on freMTPL2freq (n=50k)
  - **19× less memory** than R `smurf` package (166MB vs 3.17GB, n=20k)

- **Lasso Credibility** (`rustyglm_credibility()`): CAS Monograph 13 methodology
  - Shrinks coefficients toward a prior model (complement) instead of toward zero
  - Accepts complement as numeric vector or column name
  - `credibility_summary()` method: A/E ratio vs complement, shrinkage by coefficient
  - Mathematically exact: implemented via offset transformation `log(complement)`

### Changed
- `glm_fit_rust()` now accepts `reg_type`, `lambda`, `l1_ratio` parameters
- `rustyglm()` now supports `regularization`, `lambda`, `l1_ratio` arguments

### Benchmark results (freMTPL2freq, 678k policies)
- SMuRF Rust vs smurf R (n=50k): **10.5× faster**
- Memory at n=678k: rustyglm **478MB** vs glm() **3.74GB** (7.8× less)
- Numerical accuracy vs glm(): max difference **< 1e-5**

---

## [0.2.0] — 2026-04-03

### Added
- **Regularization support** in `glm_fit_rust()` and `rustyglm()`:
  - Ridge (L2): modified IRLS with λI diagonal term
  - Lasso (L1): coordinate descent solver (via RustyStats)
  - Elastic Net: L1 + L2 combination
- **Benchmark**: `benchmarks/bench_methods.R`
  - IRLS: 547ms | Ridge: +10% | Lasso: +35% | ElasticNet: +41%
  - All methods use ~507MB vs 3.74GB for glm() base R

### Benchmark results (regularization, n=678k)
- Ridge overhead vs IRLS: ~10% (modifies X'WX diagonal only)
- Lasso/ElasticNet overhead: ~35-41% (coordinate descent passes)
- Memory advantage maintained across all regularization types

---

## [0.1.1] — 2026-04-03

### Added
- **Parallel benchmark** (`benchmarks/bench_parallel_real.R`):
  - 22 regions sequential: rustyglm **4.7× faster** than glm()
  - CV 5-fold sequential: **4.1× faster**, **48× less RAM**
  - Parallel (parLapply): overhead dominates for short tasks on Windows
- `.gitattributes` for LF/CRLF normalization

### Fixed
- Offset not passed to Rust solver (affected all frequency models)
- Speedup table empty in benchmark output

---

## [0.1.0] — 2026-04-03

### Added
- **Phase 0**: Rust scaffold via `rextendr::use_extendr()`
  - Target: `x86_64-pc-windows-gnu` (Rtools45 compatible)
  - `hello_world()` end-to-end compilation verified

- **Phase 1**: R formula interface
  - `rustyglm(formula, data, family, offset, weights)`
  - `model.matrix()` and `model.frame()` based parsing
  - Families: poisson, gaussian, gamma, binomial, negbinomial, quasipoisson, quasibinomial
  - S3 methods: `print`, `summary`, `coef`, `fitted`, `residuals` (response/deviance/pearson), `predict`

- **Phase 2 (partial)**: Benchmarks
  - `benchmarks/bench_fremtpl2.R`: freMTPL2freq speed & memory
  - First benchmark: rustyglm **7.8× less RAM** at n=678k

### Architecture
- RustyStats core v0.6.3 as Git dependency
- extendr-api 0.8.1
- `glm_fit_rust()`: direct bridge to `fit_glm_unified()` in RustyStats

### Numerical validation
- Coefficients identical to `glm()` base R: max difference **< 1e-10** (n=200)
- Validated on freMTPL2freq (n=678k): max difference **9.52e-06**
