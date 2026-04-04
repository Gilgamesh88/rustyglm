// =============================================================================
// SMuRF — Sparse Multi-Type Regularized Feature Modeling
// Port of Devriendt et al. (2021), Insurance: Mathematics and Economics
// Original R/C++ package: https://cran.r-project.org/package=smurf
// =============================================================================

use ndarray::{Array1, Array2, s};

// =============================================================================
// Proximal operators
// =============================================================================

/// Soft threshold: sign(x) * max(|x| - gamma, 0)
#[inline]
fn soft_thresh(x: f64, gamma: f64) -> f64 {
    x.signum() * (x.abs() - gamma).max(0.0)
}

fn soft_thresh_vec(x: &Array1<f64>, gamma: f64) -> Array1<f64> {
    x.mapv(|xi| soft_thresh(xi, gamma))
}

fn soft_thresh_vec_w(x: &Array1<f64>, gammas: &Array1<f64>) -> Array1<f64> {
    Array1::from_iter(
        x.iter()
            .zip(gammas.iter())
            .map(|(&xi, &gi)| soft_thresh(xi, gi)),
    )
}

/// Group soft threshold: max(1 - gamma/||x||, 0) * x
fn group_soft_thresh(x: &Array1<f64>, gamma: f64) -> Array1<f64> {
    let norm = x.dot(x).sqrt();
    if norm <= f64::EPSILON {
        Array1::zeros(x.len())
    } else {
        x * (1.0_f64 - gamma / norm).max(0.0)
    }
}

/// Compute ADMM auxiliary matrix: (I + rho * D'D)^{-1}
/// Uses fast eigendecomposition path when all eigenvalues are non-zero.
fn make_admm_aux(
    d: usize,
    q_mat: &Array2<f64>,
    eigvals: &Array1<f64>,
    pen_mat_t: &Array2<f64>,
    pen_mat: &Array2<f64>,
    rho: f64,
    fast: bool,
) -> Array2<f64> {
    if fast {
        // Fast: I - Q * diag(rho / (1 + rho*lam)) * Q'
        let eye = Array2::<f64>::eye(d);
        let scale: Array1<f64> = eigvals.mapv(|ev| rho / (1.0 + rho * ev));
        // Q * diag(scale): multiply each column j of Q by scale[j]
        let qs: Array2<f64> =
            Array2::from_shape_fn((d, d), |(i, j)| q_mat[[i, j]] * scale[j]);
        let qt = q_mat.t().to_owned();
        eye - qs.dot(&qt)
    } else {
        // Slow: nalgebra inversion of (I + rho * D'D)
        use nalgebra::DMatrix;
        let dtd = pen_mat_t.dot(pen_mat);
        // column-major para nalgebra: vals[j*d + i] = dtd[i,j]
        let mut vals = vec![0.0f64; d * d];
        for j in 0..d {
            for i in 0..d {
                vals[j * d + i] = dtd[[i, j]];
            }
        }
        let na_mat = DMatrix::from_vec(d, d, vals);
        let na_eye = DMatrix::identity(d, d);
        let m = na_eye + na_mat * rho;
        let inv = m.try_inverse().unwrap_or_else(|| DMatrix::identity(d, d));
        Array2::from_shape_fn((d, d), |(i, j)| inv[(i, j)])
    }
}

/// ADMM proximal operator for (Generalized) Fused Lasso.
/// Direct port of admm_po_cpp from smurf C++ source.
///
/// Solves: prox_{slambda * ||D·||_1} (beta_tilde)
/// Optionally applies Group Lasso (lambda2) and sparse L1 (lambda1) components.
fn admm_proximal(
    beta_tilde: &Array1<f64>,
    slambda: f64,
    lambda1: &Array1<f64>, // sparse L1 weights (empty or [0.0] = no sparse)
    lambda2: f64,          // group L2 weight (0.0 = no group)
    pen_mat: &Array2<f64>, // D: difference matrix (m x d)
    q_mat: &Array2<f64>,   // eigenvectors of D'D (d x d)
    eigvals: &Array1<f64>, // eigenvalues of D'D
    maxiter: usize,
    beta_old: &Array1<f64>, // warm start
) -> Array1<f64> {
    let m = pen_mat.nrows();
    let d = pen_mat.ncols();
    let pen_mat_t = pen_mat.t().to_owned();

    let eps_abs = 1e-12_f64;
    let eps_rel = 1e-10_f64;
    let xi = 1.5_f64; // over-relaxation
    let mu_rho = 10.0_f64;
    let eta_rho = 2.0_f64;

    let mut rho = 1.0_f64;
    let fast = eigvals.iter().all(|&v| v.abs() >= 1e-10);
    let mut admm_aux = make_admm_aux(d, q_mat, eigvals, &pen_mat_t, pen_mat, rho, fast);

    let mut x = Array1::<f64>::zeros(d);
    let mut z_new: Array1<f64> = pen_mat.dot(beta_old); // warm start
    let mut u = Array1::<f64>::zeros(m);

    for _iter in 0..maxiter {
        let z_old = z_new.clone();

        // x-update: ADMM_aux * (beta_tilde + rho * D' * (z - u))
        let rhs = beta_tilde + &(&pen_mat_t.dot(&(&z_old - &u)) * rho);
        x = admm_aux.dot(&rhs);

        // Over-relaxation: xhat = xi * D*x + (1 - xi) * z_old
        let xhat = &pen_mat.dot(&x) * xi + &z_old * (1.0 - xi);

        // z-update: soft_thresh(xhat + u, slambda/rho)
        z_new = soft_thresh_vec(&(&xhat + &u), slambda / rho);

        // u-update
        u = &u + &xhat - &z_new;

        // Primal and dual residuals
        let primal_res = &pen_mat.dot(&x) - &z_new;
        let dual_res = &pen_mat_t.dot(&(&z_new - &z_old)) * (-rho);
        let r = primal_res.dot(&primal_res).sqrt();
        let s = dual_res.dot(&dual_res).sqrt();

        // Tolerances (Zhu 2017 adaptive rho)
        let pnx = pen_mat.dot(&x);
        let eps_p = (m as f64).sqrt() * eps_abs
            + eps_rel * pnx.dot(&pnx).sqrt().max(z_new.dot(&z_new).sqrt());
        let ptu = pen_mat_t.dot(&u);
        let eps_d = (d as f64).sqrt() * eps_abs + eps_rel * rho * ptu.dot(&ptu).sqrt();

        if r <= eps_p && s <= eps_d {
            break;
        }

        // Adapt rho
        if eps_p > 0.0 && eps_d > 0.0 {
            if r / eps_p >= mu_rho * s / eps_d {
                rho *= eta_rho;
                u = &u / eta_rho;
                admm_aux =
                    make_admm_aux(d, q_mat, eigvals, &pen_mat_t, pen_mat, rho, fast);
            } else if s / eps_d >= mu_rho * r / eps_p {
                rho /= eta_rho;
                u = &u * eta_rho;
                admm_aux =
                    make_admm_aux(d, q_mat, eigvals, &pen_mat_t, pen_mat, rho, fast);
            }
        }
    }

    // Group Lasso component (Liu et al. 2010)
    if lambda2 > 0.0 {
        x = group_soft_thresh(&x, slambda * lambda2);
    }

    // Sparse L1 component
    let has_l1 = !lambda1.is_empty() && lambda1.iter().any(|&v| v > 0.0);
    if has_l1 {
        if lambda1.len() == 1 {
            x = soft_thresh_vec(&x, slambda * lambda1[0]);
        } else {
            let thresh = lambda1.mapv(|v| slambda * v);
            x = soft_thresh_vec_w(&x, &thresh);
        }
    }

    x
}

// =============================================================================
// Per-predictor specification
// =============================================================================

pub(crate) struct PredSpec {
    pub(crate) pen_type: String,
    pub(crate) col_start: usize,
    pub(crate) col_end: usize,
    pub(crate) pen_mat: Option<Array2<f64>>,
    pub(crate) q_mat: Option<Array2<f64>>,
    pub(crate) eigvals: Option<Array1<f64>>,
    pub(crate) lambda1: Array1<f64>,
    pub(crate) lambda2: f64,
    pub(crate) group_id: i32,
}

// =============================================================================
// GLM gradient and log-likelihood
// =============================================================================

fn inv_link(eta: &Array1<f64>, family: &str) -> Array1<f64> {
    match family {
        "poisson" | "quasipoisson" | "gamma" | "negbinomial" => eta.mapv(|e| e.exp()),
        "binomial" | "quasibinomial" => eta.mapv(|e| 1.0 / (1.0 + (-e).exp())),
        _ => eta.clone(),
    }
}

/// Negative scaled log-likelihood (to minimize)
fn neg_loglik(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    weights: &Array1<f64>,
    family: &str,
) -> f64 {
    let n_eff = weights.iter().filter(|&&w| w > 0.0).count().max(1) as f64;
    let ll: f64 = y
        .iter()
        .zip(mu.iter())
        .zip(weights.iter())
        .map(|((&yi, &mui), &wi)| {
            let mui = mui.max(1e-15);
            wi * match family {
                "poisson" | "quasipoisson" => {
                    (if yi > 0.0 { yi * mui.ln() } else { 0.0 }) - mui
                }
                "gaussian" => -0.5 * (yi - mui).powi(2),
                "gamma" => -yi / mui - mui.ln(),
                "binomial" | "quasibinomial" => {
                    let mui = mui.clamp(1e-15, 1.0 - 1e-15);
                    yi * mui.ln() + (1.0 - yi) * (1.0 - mui).ln()
                }
                _ => -0.5 * (yi - mui).powi(2),
            }
        })
        .sum();
    -ll / n_eff
}

/// Gradient of negative log-likelihood w.r.t. beta: X' * (w*(mu-y)/var(mu)*dmu/deta) / n_eff
fn gradient(
    x: &Array2<f64>,
    y: &Array1<f64>,
    mu: &Array1<f64>,
    eta: &Array1<f64>,
    weights: &Array1<f64>,
    family: &str,
) -> Array1<f64> {
    let n_eff = weights.iter().filter(|&&w| w > 0.0).count().max(1) as f64;
    let resid: Array1<f64> = y
        .iter()
        .zip(mu.iter())
        .zip(eta.iter())
        .zip(weights.iter())
        .map(|(((&yi, &mui), &_etai), &wi)| {
            let mui = mui.max(1e-15);
            // (mu - y) / var(mu) * dmu/deta
            let (var_mu, dmu_deta) = match family {
                "poisson" | "quasipoisson" => (mui, mui),
                "gaussian" => (1.0, 1.0),
                "gamma" => (mui * mui, mui),
                "binomial" | "quasibinomial" => {
                    let v = mui * (1.0 - mui);
                    (v, v)
                }
                _ => (1.0, 1.0),
            };
            wi * (mui - yi) / var_mu * dmu_deta
        })
        .collect::<Vec<_>>()
        .into();
    x.t().dot(&resid) / n_eff
}

// =============================================================================
// Total penalty value (for objective function)
// =============================================================================

fn total_penalty(beta: &Array1<f64>, specs: &[PredSpec], lambda: f64) -> f64 {
    specs
        .iter()
        .map(|sp| {
            let b = beta.slice(s![sp.col_start..sp.col_end]).to_owned();
            match sp.pen_type.as_str() {
                "lasso" => lambda * b.mapv(|v| v.abs()).sum(),
                "grouplasso" => lambda * b.dot(&b).sqrt(),
                "flasso" | "gflasso" => {
                    let mut pen = 0.0;
                    if let Some(ref dm) = sp.pen_mat {
                        let db = dm.dot(&b);
                        pen += lambda * db.mapv(|v| v.abs()).sum();
                    }
                    if sp.lambda1.iter().any(|&v| v > 0.0) {
                        pen += lambda
                            * b.iter()
                                .zip(sp.lambda1.iter())
                                .map(|(bi, li)| li * bi.abs())
                                .sum::<f64>();
                    }
                    if sp.lambda2 > 0.0 {
                        pen += lambda * sp.lambda2 * b.dot(&b).sqrt();
                    }
                    pen
                }
                _ => 0.0,
            }
        })
        .sum()
}

// =============================================================================
// Apply per-predictor proximal operators
// =============================================================================

fn apply_proximal(
    beta_tilde: &Array1<f64>,
    beta_old: &Array1<f64>,
    specs: &[PredSpec],
    lambda: f64,
    step: f64,
    group_norms: &[f64],
) -> Array1<f64> {
    let mut out = beta_tilde.clone();
    let slambda = lambda * step;

    for (j, sp) in specs.iter().enumerate() {
        let s = sp.col_start;
        let e = sp.col_end;
        let bt = beta_tilde.slice(s![s..e]).to_owned();
        let bo = beta_old.slice(s![s..e]).to_owned();

        let prox: Array1<f64> = match sp.pen_type.as_str() {
            "none" => bt,

            "lasso" => soft_thresh_vec(&bt, slambda),

            "grouplasso" => {
                let norm = group_norms[j].max(f64::EPSILON);
                &bt * (1.0_f64 - slambda / norm).max(0.0)
            }

            "flasso" | "gflasso" => match (&sp.pen_mat, &sp.q_mat, &sp.eigvals) {
                (Some(dm), Some(qm), Some(ev)) => {
                    admm_proximal(&bt, slambda, &sp.lambda1, sp.lambda2, dm, qm, ev, 10_000, &bo)
                }
                _ => bt,
            },

            _ => bt,
        };

        out.slice_mut(s![s..e]).assign(&prox);
    }
    out
}

// =============================================================================
// FISTA outer loop
// =============================================================================

struct SmurfResult {
    coefficients: Vec<f64>,
    iterations: i32,
    converged: bool,
    final_objective: f64,
}

#[allow(clippy::too_many_arguments)]
fn fista_loop(
    x_mat: &Array2<f64>,
    y: &Array1<f64>,
    offset: &Array1<f64>,
    weights: &Array1<f64>,
    family: &str,
    specs: &[PredSpec],
    lambda: f64,
    maxiter: usize,
    epsilon: f64,
    step_init: f64,
    tau: f64,
    start: &Array1<f64>,
) -> SmurfResult {
    let mut beta_old = start.clone();
    let mut beta_new = start.clone();
    let mut theta = start.clone();

    let mut alpha_old = 0.0_f64;
    let mut alpha_new = 1.0_f64;
    let step_low = 1e-14_f64;
    let mut step = step_init;
    let mut subs_restart = 0usize;
    let mut iter = 0usize;

    // Initial objective
    let mu0 = inv_link(&(x_mat.dot(start) + offset), family);
    let mut obj_old = 0.0_f64; // triggers first iteration
    let mut obj_new = neg_loglik(y, &mu0, weights, family) + total_penalty(start, specs, lambda);

    while (obj_old == 0.0 || (obj_old - obj_new).abs() / obj_old.abs().max(1e-30) > epsilon)
        && iter < maxiter
    {
        obj_old = obj_new;
        beta_old = beta_new.clone();

        // Gradient at theta
        let eta_th = x_mat.dot(&theta) + offset;
        let mu_th = inv_link(&eta_th, family);
        let f_theta = neg_loglik(y, &mu_th, weights, family);
        let grad = gradient(x_mat, y, &mu_th, &eta_th, weights, family);

        // Gradient step
        let beta_tilde = &theta - &(&grad * step);

        // Group norms for GroupLasso
        let group_norms: Vec<f64> = specs
            .iter()
            .map(|sp| {
                let b = beta_tilde.slice(s![sp.col_start..sp.col_end]).to_owned();
                b.dot(&b).sqrt()
            })
            .collect();

        // Proximal step
        let mut beta_cand =
            apply_proximal(&beta_tilde, &beta_old, specs, lambda, step, &group_norms);

        // Evaluate
        let mu_cand = inv_link(&(x_mat.dot(&beta_cand) + offset), family);
        let f_cand = neg_loglik(y, &mu_cand, weights, family);
        let g_cand = total_penalty(&beta_cand, specs, lambda);
        let mut obj_cand = f_cand + g_cand;

        // Backtracking (Armijo condition)
        let diff = &beta_cand - &theta;
        let mut h_theta =
            f_theta + grad.dot(&diff) + diff.dot(&diff) / (2.0 * step) + g_cand;

        while obj_cand > h_theta && step >= step_low {
            step *= tau;
            let bt2 = &theta - &(&grad * step);

            let gn2: Vec<f64> = specs
                .iter()
                .map(|sp| {
                    let b = bt2.slice(s![sp.col_start..sp.col_end]).to_owned();
                    b.dot(&b).sqrt()
                })
                .collect();

            beta_cand = apply_proximal(&bt2, &beta_old, specs, lambda, step, &gn2);

            let mu_c = inv_link(&(x_mat.dot(&beta_cand) + offset), family);
            let f_c = neg_loglik(y, &mu_c, weights, family);
            let g_c = total_penalty(&beta_cand, specs, lambda);
            obj_cand = f_c + g_c;

            let d2 = &beta_cand - &theta;
            h_theta = f_theta + grad.dot(&d2) + d2.dot(&d2) / (2.0 * step) + g_c;
        }

        beta_new = beta_cand;
        obj_new = obj_cand;

        // FISTA acceleration (Nesterov)
        alpha_old = alpha_new;
        alpha_new = (1.0 + (1.0 + 4.0 * alpha_old * alpha_old).sqrt()) / 2.0;

        // Monotone restart if objective increases
        if obj_new > obj_old * (1.0 + epsilon) {
            beta_new = beta_old.clone();
            obj_new = obj_old;
            obj_old = 0.0;
            alpha_old = 0.0;
            alpha_new = 1.0;
            subs_restart += 1;
        } else {
            subs_restart = 0;
        }

        if subs_restart > 1 {
            break;
        }

        // Theta update (momentum)
        let accel = (alpha_old - 1.0) / alpha_new;
        theta = &beta_new + &((&beta_new - &beta_old) * accel);

        iter += 1;
    }

    let converged = iter < maxiter && subs_restart <= 1 && step >= step_low;

    SmurfResult {
        coefficients: beta_new.to_vec(),
        iterations: iter as i32,
        converged,
        final_objective: obj_new,
    }
}

// =============================================================================
// Helper: reconstruct Array2 from flat column-major data
// =============================================================================

fn flat_to_array2(data: &[f64], nrows: usize, ncols: usize) -> Array2<f64> {
    // R stores column-major; convert to row-major for ndarray
    let mut arr = Array2::zeros((nrows, ncols));
    for col in 0..ncols {
        for row in 0..nrows {
            arr[[row, col]] = data[col * nrows + row];
        }
    }
    arr
}

// =============================================================================
// extendr interface
// =============================================================================

/// Fit SMuRF GLM from R using the Rust FISTA implementation.
///
/// @param response     Response vector (length n).
/// @param x_flat       Design matrix, column-major (length n*p).
/// @param nrows        n (number of observations).
/// @param ncols        p (number of columns in design matrix).
/// @param family       Distribution family string.
/// @param offset       Offset vector (length n; zeros if none).
/// @param weights      Prior weights (length n; ones if none).
/// @param pen_types    Penalty type per predictor block.
/// @param col_starts   0-based start column per predictor (inclusive).
/// @param col_ends     0-based end column per predictor (exclusive).
/// @param pen_mats_flat Concatenated D matrices (column-major).
/// @param pen_mat_nrows Number of rows in each D matrix.
/// @param pen_mat_ncols Number of cols in each D matrix.
/// @param q_mats_flat  Concatenated eigenvector matrices of D'D.
/// @param q_mat_nrows  Rows of each Q.
/// @param q_mat_ncols  Cols of each Q.
/// @param eigvals_flat Concatenated eigenvalue vectors.
/// @param eigval_lens  Length of each eigenvalue vector.
/// @param lambda1_flat Concatenated sparse L1 weight vectors.
/// @param lambda1_lens Length of each lambda1 vector.
/// @param lambda2s     Group L2 weight per predictor.
/// @param group_ids    Group ID (0 = none, >0 = cross-predictor group).
/// @param lambda       Global penalty parameter.
/// @param maxiter      Maximum FISTA iterations.
/// @param epsilon      Convergence tolerance.
/// @param step_init    Initial step size for backtracking.
/// @param tau          Step size reduction factor (0 < tau < 1).
/// @param start        Starting values for beta (length p; zeros if empty).
///

pub fn run_smurf(
    x_mat: &Array2<f64>,
    y: &Array1<f64>,
    off: &Array1<f64>,
    wts: &Array1<f64>,
    family: &str,
    specs: Vec<PredSpec>,
    lambda: f64,
    maxiter: usize,
    epsilon: f64,
    step_init: f64,
    tau: f64,
    start: &Array1<f64>,
) -> (Vec<f64>, i32, bool, f64) {
    let res = fista_loop(x_mat, y, off, wts, family, &specs,
                         lambda, maxiter, epsilon, step_init, tau, start);
    (res.coefficients, res.iterations, res.converged, res.final_objective)
}

pub fn build_specs(
    pen_types: &[String],
    col_starts: &[i32],
    col_ends: &[i32],
    pen_mats_flat: &[f64],
    pen_mat_nrows: &[i32],
    pen_mat_ncols: &[i32],
    q_mats_flat: &[f64],
    q_mat_nrows: &[i32],
    q_mat_ncols: &[i32],
    eigvals_flat: &[f64],
    eigval_lens: &[i32],
    lambda1_flat: &[f64],
    lambda1_lens: &[i32],
    lambda2s: &[f64],
    group_ids: &[i32],
) -> Vec<PredSpec> {
    let n_preds = pen_types.len();
    let mut pm_off = 0usize;
    let mut qm_off = 0usize;
    let mut ev_off = 0usize;
    let mut l1_off = 0usize;

    (0..n_preds).map(|j| {
        let mr = pen_mat_nrows[j] as usize;
        let mc = pen_mat_ncols[j] as usize;
        let pen_mat = if mr > 0 && mc > 0 {
            let data = &pen_mats_flat[pm_off..pm_off + mr * mc];
            pm_off += mr * mc;
            Some(flat_to_array2(data, mr, mc))
        } else { None };

        let qr = q_mat_nrows[j] as usize;
        let qc = q_mat_ncols[j] as usize;
        let q_mat = if qr > 0 && qc > 0 {
            let data = &q_mats_flat[qm_off..qm_off + qr * qc];
            qm_off += qr * qc;
            Some(flat_to_array2(data, qr, qc))
        } else { None };

        let el = eigval_lens[j] as usize;
        let eigvals = if el > 0 {
            let data = eigvals_flat[ev_off..ev_off + el].to_vec();
            ev_off += el;
            Some(Array1::from_vec(data))
        } else { None };

        let ll = lambda1_lens[j] as usize;
        let lambda1 = if ll > 0 {
            let data = lambda1_flat[l1_off..l1_off + ll].to_vec();
            l1_off += ll;
            Array1::from_vec(data)
        } else {
            Array1::from_vec(vec![0.0])
        };

        PredSpec {
            pen_type:  pen_types[j].clone(),
            col_start: col_starts[j] as usize,
            col_end:   col_ends[j] as usize,
            pen_mat,
            q_mat,
            eigvals,
            lambda1,
            lambda2:  lambda2s[j],
            group_id: group_ids[j],
        }
    }).collect()
}