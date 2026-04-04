use extendr_api::prelude::*;
use rustystats_core::{
    fit_glm_unified,
    FitConfig, RegularizationConfig,
    families::{
        PoissonFamily, GaussianFamily, GammaFamily,
        BinomialFamily, NegativeBinomialFamily,
    },
    links::{LogLink, IdentityLink, LogitLink},
};
use ndarray::{Array1, Array2};

mod smurf;

/// Ajusta un GLM desde R usando RustyStats core.
/// @export
#[extendr]
fn glm_fit_rust(
    response: &[f64],
    x_flat:   &[f64],
    nrows:    i32,
    ncols:    i32,
    family:   &str,
    offset:   &[f64],
    reg_type: &str,
    lambda:   f64,
    l1_ratio: f64,
) -> Vec<f64> {
    let n = nrows as usize;
    let p = ncols as usize;

    let mut x_rm = vec![0.0f64; n * p];
    for j in 0..p {
        for i in 0..n {
            x_rm[i * p + j] = x_flat[j * n + i];
        }
    }
    let x = match Array2::from_shape_vec((n, p), x_rm) {
        Ok(m)  => m,
        Err(e) => { rprintln!("Error construyendo matriz: {}", e); return vec![]; }
    };

    let y = Array1::from_vec(response.to_vec());
    let offset_opt = if offset.is_empty() || offset.iter().all(|&v| v == 0.0) {
        None
    } else {
        Some(Array1::from_vec(offset.to_vec()))
    };

    let reg = match reg_type {
        "ridge"      => RegularizationConfig::ridge(lambda),
        "lasso"      => RegularizationConfig::lasso(lambda),
        "elasticnet" => RegularizationConfig::elastic_net(lambda, l1_ratio),
        _            => RegularizationConfig::none(),
    };

    let mut config = FitConfig::default();
    config.regularization = reg;

    let result = match family {
        "gaussian" =>
            fit_glm_unified(&y, x.view(), &GaussianFamily, &IdentityLink,
                            &config, offset_opt.as_ref(), None, None),
        "gamma" =>
            fit_glm_unified(&y, x.view(), &GammaFamily, &LogLink,
                            &config, offset_opt.as_ref(), None, None),
        "binomial" | "quasibinomial" =>
            fit_glm_unified(&y, x.view(), &BinomialFamily, &LogitLink,
                            &config, offset_opt.as_ref(), None, None),
        "negbinomial" =>
            fit_glm_unified(&y, x.view(), &NegativeBinomialFamily::default(), &LogLink,
                            &config, offset_opt.as_ref(), None, None),
        _ =>
            fit_glm_unified(&y, x.view(), &PoissonFamily, &LogLink,
                            &config, offset_opt.as_ref(), None, None),
    };

    match result {
        Ok(r)  => r.coefficients.to_vec(),
        Err(e) => { rprintln!("Error GLM Rust: {}", e); vec![] }
    }
}

/// Ajusta un GLM con penalizaciones multi-tipo (SMuRF — FISTA Rust).
/// @export
#[extendr]
#[allow(clippy::too_many_arguments)]
fn smurf_fit_rust(
    response:      &[f64],
    x_flat:        &[f64],
    nrows:         i32,
    ncols:         i32,
    family:        &str,
    offset:        &[f64],
    weights:       &[f64],
    pen_types:     Vec<String>,
    col_starts:    Vec<i32>,
    col_ends:      Vec<i32>,
    pen_mats_flat: &[f64],
    pen_mat_nrows: Vec<i32>,
    pen_mat_ncols: Vec<i32>,
    q_mats_flat:   &[f64],
    q_mat_nrows:   Vec<i32>,
    q_mat_ncols:   Vec<i32>,
    eigvals_flat:  &[f64],
    eigval_lens:   Vec<i32>,
    lambda1_flat:  &[f64],
    lambda1_lens:  Vec<i32>,
    lambda2s:      Vec<f64>,
    group_ids:     Vec<i32>,
    lambda:        f64,
    maxiter:       i32,
    epsilon:       f64,
    step_init:     f64,
    tau:           f64,
    start:         &[f64],
) -> List {
    let n = nrows as usize;
    let p = ncols as usize;

    let mut x_rm = vec![0.0f64; n * p];
    for j in 0..p {
        for i in 0..n {
            x_rm[i * p + j] = x_flat[j * n + i];
        }
    }
    let x_mat = match Array2::from_shape_vec((n, p), x_rm) {
        Ok(m)  => m,
        Err(e) => {
            rprintln!("Error X: {}", e);
            return list!(coefficients = Vec::<f64>::new(),
                         iterations = 0_i32, converged = false, objective = 0.0_f64);
        }
    };

    let y   = Array1::from_vec(response.to_vec());
    let off = if offset.is_empty()  { Array1::zeros(n) }
              else { Array1::from_vec(offset.to_vec()) };
    let wts = if weights.is_empty() { Array1::ones(n) }
              else { Array1::from_vec(weights.to_vec()) };
    let b0  = if start.is_empty()   { Array1::zeros(p) }
              else { Array1::from_vec(start.to_vec()) };

    let specs = smurf::build_specs(
        &pen_types, &col_starts, &col_ends,
        pen_mats_flat, &pen_mat_nrows, &pen_mat_ncols,
        q_mats_flat,   &q_mat_nrows,   &q_mat_ncols,
        eigvals_flat,  &eigval_lens,
        lambda1_flat,  &lambda1_lens,
        &lambda2s, &group_ids,
    );

    let (coefs, iters, conv, obj) = smurf::run_smurf(
        &x_mat, &y, &off, &wts, family, specs,
        lambda, maxiter as usize, epsilon, step_init, tau, &b0,
    );

    list!(
        coefficients = coefs,
        iterations   = iters,
        converged    = conv,
        objective    = obj
    )
}

extendr_module! {
    mod rustyglm;
    fn glm_fit_rust;
    fn smurf_fit_rust;
}
