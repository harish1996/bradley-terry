from typing import Tuple, Protocol, Optional, Union
import numpy as np
from pydantic import BaseModel


class OptimizableModel(Protocol):
    def gradient(self) -> np.ndarray:
        """Compute the gradient vector."""
        ...

    def hessian(self) -> np.ndarray:
        """Compute the Hessian matrix."""
        ...

    def adjust_params(self, delta: np.ndarray) -> None:
        """Adjust the model parameters based on delta."""
        ...


def newton_raphson(
    model: OptimizableModel,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Perform the Newton-Raphson optimization method on a given model.

    Args:
        model (OptimizableModel): The model to optimize.
        max_iter (int): Maximum number of iterations. Default is 100.
        tol (float): Convergence tolerance for gradient norm. Default is 1e-6.
        verbose (bool): If True, print details of each iteration. Default is False.

    Returns:
        Tuple[np.ndarray, int]: Optimized parameters and the number of iterations performed.
    """
    for iteration in range(max_iter):
        grad = model.gradient()
        hess = model.hessian()

        # Check if Hessian is singular
        try:
            delta = np.linalg.solve(hess, grad)  # Solve Hessian * delta = gradient
        except np.linalg.LinAlgError:
            raise ValueError("Hessian is singular; Newton-Raphson cannot proceed.")

        # Update parameters
        model.adjust_params(-delta)

        # Verbose logging
        if verbose:
            print(f"Iteration {iteration + 1}:")
            print(f"  Gradient Norm: {np.linalg.norm(grad):.6f}")
            print(f"  Parameter Update: {delta}")
            print(f"  Updated Parameters: {model.params}\n")

        # Check convergence
        if np.linalg.norm(grad) < tol:
            if verbose:
                print("Convergence achieved.")
            return model.params, iteration + 1

    if verbose:
        print("Newton-Raphson did not converge within the maximum number of iterations.")
    raise ValueError("Newton-Raphson did not converge within the maximum number of iterations.")


class BradleyTerryModel(BaseModel):
    M: np.ndarray  # Pairwise comparison matrix
    params: np.ndarray  # Parameters p_2, p_3, ..., p_n
    a: float  # Scaling factor for the probability exponent

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        M: np.ndarray,
        params: Optional[Union[float, str, np.ndarray]] = None,
        a: Optional[Union[float, str]] = None,
    ):
        """
        Initialize the Bradley-Terry Model.
        """
        n = M.shape[0]  # Number of items
        if params is None:
            params_array = np.zeros(n - 1)  # Default: all zeros
        elif isinstance(params, float):
            params_array = np.full(n - 1, params)
        elif isinstance(params, str) and params.lower() == "random":
            params_array = np.random.rand(n - 1)
        elif isinstance(params, np.ndarray):
            if params.shape == (n - 1,):
                params_array = params
            else:
                raise ValueError("params must have shape (n-1,) to match the number of items.")
        else:
            raise ValueError("Invalid params initialization.")

        if a is None:
            a_value = 1.0
        elif isinstance(a, float):
            a_value = a
        elif isinstance(a, str) and a.endswith("%"):
            try:
                percentage = float(a.strip('%')) / 100.0
                a_value = np.log(1 + percentage)
            except ValueError:
                raise ValueError("Invalid percentage format for 'a'. Use a float or '10%' format.")
        else:
            raise ValueError("Invalid 'a' initialization.")

        super().__init__(M=M, params=params_array, a=a_value)

    def gradient(self) -> np.ndarray:
        n = self.M.shape[0]
        grad = np.zeros(n - 1)
        probs = np.zeros((n, n))
        exp_params = np.exp(np.insert(self.params, 0, 0) * self.a)

        for i in range(n):
            for j in range(n):
                if i != j:
                    probs[i, j] = exp_params[i] / (exp_params[i] + exp_params[j])

        for i in range(1, n):
            grad[i - 1] = self.a * sum(
                self.M[i, j] - ((self.M[j, i] + self.M[i, j]) * probs[i, j]) for j in range(n) if j != i
            )

        return grad

    def hessian(self) -> np.ndarray:
        n = self.M.shape[0]
        hess = np.zeros((n - 1, n - 1))
        probs = np.zeros((n, n))
        exp_params = np.exp(np.insert(self.params, 0, 0) * self.a)

        for i in range(n):
            for j in range(n):
                if i != j:
                    probs[i, j] = exp_params[i] / (exp_params[i] + exp_params[j])

        for i in range(1, n):
            for j in range(1, n):
                if j == i:
                    hess[i - 1, i - 1] = -(self.a**2) * sum(
                        (self.M[i, k] + self.M[k, i]) * (probs[i, k] * probs[k, i]) for k in range(n) if k != i
                    )
                else:
                    hess[i - 1, j - 1] = (self.a**2) * ((self.M[i, j] + self.M[j, i]) * probs[i, j] * probs[j, i])
        return hess

    def adjust_params(self, delta: np.ndarray) -> None:
        self.params += delta

# win_matrix = x = np.random.random_integers(0,250,(100,100))
# for row, y in enumerate(x):
#     win_matrix[row][row] = 0
win_matrix = np.array([[0, 46, 93, 100, 102, 239, 202, 92, 8, 194],
 [52, 0, 201, 81, 151, 194, 177, 210, 204, 149],
 [172, 163, 0, 221, 69, 55, 215, 19, 86, 233],
 [68, 131, 66, 0, 193, 29, 145, 36, 45, 201],
 [157, 9, 38, 34, 0, 161, 131, 127, 28, 203],
 [120, 15, 77, 163, 193, 0, 204, 157, 113, 165],
 [26, 171, 246, 196, 48, 67, 0, 116, 39, 130],
 [18, 117, 232, 216, 9, 122, 158, 0, 114, 127],
 [155, 181, 83, 4, 210, 23, 179, 222, 0, 120],
 [207, 188, 72, 216, 87, 201, 166, 109, 31, 0]])
print(win_matrix)

model = BradleyTerryModel(M=win_matrix, params="random", a="1%")
optimized_params, iterations = newton_raphson(model, max_iter=100, tol=1e-6, verbose=True)
model.M[0][1] += 300
optimized_params, iterations = newton_raphson(model, max_iter=100, tol=1e-6, verbose=True)
