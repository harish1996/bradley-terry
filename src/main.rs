use ndarray::{s,Array1, Array2};
use rand::prelude::*;
use std::f64;

pub mod newton_raphson {

    use ndarray_linalg::{Solve,Norm};

    pub trait OptimizableModel {
        fn gradient(&self) -> ndarray::Array1<f64>;
        fn hessian(&self) -> ndarray::Array2<f64>;
        fn adjust_params(&mut self, delta: &ndarray::Array1<f64>);
        fn params(&self) -> ndarray::Array1<f64>;
    }

    pub fn optimize<M: OptimizableModel>(
        model: &mut M,
        max_iter: usize,
        tol: f64,
        verbose: bool,
    ) -> Result<(ndarray::Array1<f64>, usize), Box<dyn std::error::Error>> {
        for iteration in 0..max_iter {
            let grad = model.gradient();
            let hess = model.hessian();

            let delta = hess.solve(&grad)?;
            model.adjust_params(&(-delta.clone()));

            if verbose {
                println!("Iteration {}:", iteration + 1);
                println!("  Gradient Norm: {:.6}", grad.norm());
                println!("  Parameter Update: {:?}", delta);
                println!("  Updated Parameters: {:?}\n", model.params());
            }

            if grad.norm() < tol {
                if verbose {
                    println!("Convergence achieved.");
                }
                return Ok((model.params(), iteration + 1));
            }
        }

        Err("Newton-Raphson did not converge within the maximum number of iterations.".into())
    }
}

pub mod bradley_terry {
    use super::*;

    pub struct BradleyTerryModel {
        pub pairwise_matrix: Array2<f64>,
        pub params: Array1<f64>,
        pub scaling_factor: f64,
    }

    impl BradleyTerryModel {
        pub fn new(
            pairwise_matrix: Array2<f64>,
            params: Option<&str>,
            scaling_factor: Option<&str>,
        ) -> Self {
            let n = pairwise_matrix.shape()[0];
            let params = match params {
                Some("random") => Array1::from_iter((0..n - 1).map(|_| rand::random::<f64>())),
                _ => Array1::zeros(n - 1),
            };

            let scaling_factor = match scaling_factor {
                Some(value) if value.ends_with('%') => {
                    let percentage = value.trim_end_matches('%').parse::<f64>().unwrap();
                    (1.0 + percentage / 100.0).ln()
                }
                _ => 1.0,
            };

            Self {
                pairwise_matrix,
                params,
                scaling_factor,
            }
        }

        fn compute_probs(&self) -> Array2<f64> {
            let n = self.pairwise_matrix.shape()[0];
            let mut exp_params = Array1::zeros(n);
            exp_params.slice_mut(s![1..]).assign(&(self.params.mapv(|x| (x * self.scaling_factor).exp())));
            exp_params[0] = 1.0;

            let mut probs = Array2::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        probs[[i, j]] = exp_params[i] / (exp_params[i] + exp_params[j]);
                    }
                }
            }
            probs
        }
    }

    impl newton_raphson::OptimizableModel for BradleyTerryModel {
        fn gradient(&self) -> Array1<f64> {
            let n = self.pairwise_matrix.shape()[0];
            let probs = self.compute_probs();
            let mut grad = Array1::zeros(n - 1);

            for i in 1..n {
                grad[i - 1] = self.scaling_factor
                    * (0..n)
                        .filter(|&j| j != i)
                        .map(|j| {
                            self.pairwise_matrix[[i, j]]
                                - ((self.pairwise_matrix[[i, j]] + self.pairwise_matrix[[j, i]])
                                    * probs[[i, j]])
                        })
                        .sum::<f64>();
            }
            grad
        }

        fn hessian(&self) -> Array2<f64> {
            let n = self.pairwise_matrix.shape()[0];
            let probs = self.compute_probs();
            let mut hess = Array2::zeros((n - 1, n - 1));

            for i in 1..n {
                for j in 1..n {
                    if i == j {
                        hess[[i - 1, j - 1]] = -(self.scaling_factor.powi(2))
                            * (0..n)
                                .filter(|&k| k != i)
                                .map(|k| {
                                    (self.pairwise_matrix[[i, k]] + self.pairwise_matrix[[k, i]])
                                        * probs[[i, k]]
                                        * (1.0 - probs[[i, k]])
                                })
                                .sum::<f64>();
                    } else {
                        hess[[i - 1, j - 1]] = self.scaling_factor.powi(2)
                            * ( self.pairwise_matrix[[i, j]] + self.pairwise_matrix[[j,i]] )
                            * probs[[i, j]]
                            * (1.0 - probs[[i, j]]);
                    }
                }
            }
            hess
        }

        fn adjust_params(&mut self, delta: &Array1<f64>) {
            self.params += delta;
        }

        fn params(&self) -> Array1<f64> {
            self.params.clone()
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 10;
    let mut pairwise_matrix = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if i != j {
                pairwise_matrix[[i, j]] = rand::thread_rng().gen_range(0..250) as f64;
            }
        }
    }

    println!("Pairwise matrix: {}", pairwise_matrix);
    let mut model = bradley_terry::BradleyTerryModel::new(pairwise_matrix, Some("random"), Some("10%"));

    let (optimized_params, iterations) = newton_raphson::optimize(&mut model, 100, 1e-6, true)?;

    println!("Optimized Parameters: {:?}", optimized_params);
    println!("Iterations: {}", iterations);

    Ok(())
}
