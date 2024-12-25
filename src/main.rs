use ndarray::{s, Array1, Array2};
use rand::prelude::*;
use std::f64;

pub mod newton_raphson {

    use ndarray_linalg::{Norm, Solve};

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
                println!("Gradient: {}", grad);
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

pub mod polynomial {
    use super::*;

    #[derive(Clone)]
    pub struct BradleyTerryModel {
        pub pairwise_matrix: Array2<f64>,
        pub params: Array1<f64>,
        pub a: f64,
        pub b: f64,
        pub c: f64,
        pub d: f64,
    }

    impl BradleyTerryModel {
        // Constructor for the Bradley-Terry model
        pub fn new(pairwise_matrix: Array2<f64>, params: Option<&str>, a: f64, b: f64, c: f64, d: f64) -> Self {
            let n = pairwise_matrix.shape()[0];
            let mut params = match params {
                Some("random") => Array1::from_iter((0..n).map(|_| rand::random::<f64>())),
                _ => Array1::zeros(n),
            };
            params[0] = 0.0;
            Self {
                pairwise_matrix,
                params,
                a,
                b,
                c,
                d,
            }
        }

        // Function f(x) = ( (ax - b) + sqrt( (ax-b)^2 + c ) )^d
        fn f(&self, x: f64) -> f64 {
            let ax_minus_b = self.a * x - self.b;
            (ax_minus_b + ((ax_minus_b.powi(2) + self.c).sqrt())).powf(self.d)
        }

        // Derivative of f(x) with respect to x
        fn f_prime(&self, x: f64) -> f64 {
            let ax_minus_b = self.a * x - self.b;
            let denom = (ax_minus_b + (ax_minus_b.powi(2) + self.c).sqrt()).powf(self.d - 1.0);
            self.a * denom * (1.0 + self.d * (ax_minus_b.powi(2) + self.c).sqrt()).powf(1.0)
        }

        // Second derivative of f(x) with respect to x
        fn f_double_prime(&self, x: f64) -> f64 {
            let ax_minus_b = self.a * x - self.b;
            let denom = (ax_minus_b + (ax_minus_b.powi(2) + self.c).sqrt()).powf(self.d - 1.0);
            let term1 = 2.0 * self.d * self.a * ax_minus_b * denom;
            term1 + self.d * self.d * (ax_minus_b.powi(2) + self.c).sqrt().powi(2)
        }

        // Compute the probabilities P(x_i, x_j) using the modified function
        fn compute_probs(&self) -> Array2<f64> {
            let n = self.pairwise_matrix.shape()[0];
            let mut probs = Array2::zeros((n, n));

            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        let f_i = self.f(self.params[i]);
                        let f_j = self.f(self.params[j]);
                        probs[[i, j]] = f_i / (f_i + f_j);
                    }
                }
            }
            probs
        }
    }

    impl newton_raphson::OptimizableModel for BradleyTerryModel {
        fn gradient(&self) -> Array1<f64> {
            let n = self.pairwise_matrix.shape()[0];
            let mut grad = Array1::zeros(n-1);
    
            // Iterate through each parameter x_i
            for i in 1..n {
                let mut grad_i = 0.0;
    
                // Compute f(x_i) and f'(x_i) for the current i
                let f_i = self.f(self.params[i]);
                let f_prime_i = self.f_prime(self.params[i]);
    
                // Sum over all pairwise comparisons for the gradient formula
                for j in 0..n {
                    if i != j {
                        let f_j = self.f(self.params[j]);
                        let denominator = f_i + f_j;
                        let term_1 = ( f_j / f_i ) * self.pairwise_matrix[[i,j]];
                        let term_2 = self.pairwise_matrix[[j,i]];
    
                        // The summation term X_ij * f(x_j) / (f(x_i) + f(x_j))
                        grad_i += (term_1 - term_2) / denominator;
                    }
                }
    
                // Multiply by f'(x_i) / f(x_i) for the final gradient value
                grad[i-1] = f_prime_i * grad_i;
            }
    
            grad
        }

        fn hessian(&self) -> Array2<f64> {
            let n = self.pairwise_matrix.shape()[0];
            let mut hess = Array2::zeros((n-1, n-1));

            for i in 1..n {
                let f_i = self.f(self.params[i]);
                let f_prime_i = self.f_prime(self.params[i]);
                let f_double_prime_i = self.f_double_prime(self.params[i]);
    
                for j in 1..n {
                    if i == j {
                        // First summation term: (f''(x_i) * f(x_i) - f'(x_i)^2) / f(x_i)^2
                        let first_term = f_double_prime_i;
                        // Second summation term: f'(x_i)^2 / f(x_i)
                        let second_term = f_prime_i.powi(2) / f_i.powi(2);
                        let mut first_summation = 0.0;
                        let mut second_summation = 0.0;
                        for k in 0..n {
                            if k != i {
                                let f_k = self.f(self.params[k]);
                                let a = self.pairwise_matrix[[i,k]];
                                let b = self.pairwise_matrix[[k,i]];
                                let denominator = f_i + f_k;
                                first_summation += ( (f_k * a) / f_i ) - b;
                                second_summation += ((b*f_i*f_i) - (2.0*a*f_i*f_k) - (a*f_k*f_k)) / denominator.powi(2);
                            }
                        }
                        hess[[i-1, i-1]] = ( first_term * first_summation ) + ( second_term * second_summation );
                    } else {
                        let f_j = self.f(self.params[j]);
                        let f_prime_j = self.f_prime(self.params[j]);
                        hess[[i-1, j-1]] = ( f_prime_i * f_prime_j * ( self.pairwise_matrix[[i,j]] + self.pairwise_matrix[[j,i]] ) ) / ( f_i + f_j ).powi(2);
                    }
                }
            }
            hess
        }

        fn adjust_params(&mut self, delta: &Array1<f64>) {
            for i in 1..delta.len()+1 {
                self.params[i] += delta[i-1];
            }
        }

        fn params(&self) -> Array1<f64> {
            self.params.clone()
        }
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
            exp_params
                .slice_mut(s![1..])
                .assign(&(self.params.mapv(|x| (x * self.scaling_factor).exp())));
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
                            * (self.pairwise_matrix[[i, j]] + self.pairwise_matrix[[j, i]])
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
    let n = 5;
    let mut pairwise_matrix = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if i != j {
                pairwise_matrix[[i, j]] = rand::thread_rng().gen_range(0..250) as f64;
            }
        }
    }

    println!("Pairwise matrix: {}", pairwise_matrix);
    // let mut model =
        // bradley_terry::BradleyTerryModel::new(pairwise_matrix, Some("random"), Some("10%"));

    let mut model = polynomial::BradleyTerryModel::new(pairwise_matrix.clone(), Some("random"), 1.0, 0.0, 1.0, 1.0);
    let (optimized_params, iterations) = newton_raphson::optimize(&mut model, 100, 1e-6, true)?;

    println!("Optimized Parameters: {:?}", optimized_params);
    println!("Iterations: {}", iterations);

    let mut model_2 = bradley_terry::BradleyTerryModel::new(pairwise_matrix, Some("random"), Some("10%"));
    let ( optimized_params, iterations ) = newton_raphson::optimize(&mut model_2, 100, 1e-6, true)?;

    println!("Optimized Parameters: {:?}", optimized_params);
    println!("Iterations: {}", iterations);

    Ok(())
}
