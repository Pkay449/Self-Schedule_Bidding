import numpy as np
from scipy.optimize import minimize_scalar
from qpsolvers import solve_qp

def VRx_weights(phi, Y, weights_lsqlin):
    """
    Compute the weights to approximate a state using quadratic programming.

    Parameters:
    - phi: 2D array (N x M), each row represents a state, each column a feature.
    - Y: 1D array (M,), the target state.
    - weights_lsqlin: 1D array (M,), feature scaling weights.

    Returns:
    - weights: 1D array (N,), optimized weights.
    """
    if Y.ndim > 1:
        Y = Y.flatten()

    # Step 1: Ignore constant features
    mask = np.std(phi, axis=0) < 0.1
    weights_lsqlin[mask] = 0

    # Step 2: Compute weighted Euclidean distance
    diff = (phi - Y) * weights_lsqlin
    dist = np.sqrt(np.sum(diff**2, axis=1))

    N = phi.shape[0]

    def kernel_dist(h_kernel):
        return np.exp(-((dist / h_kernel) ** 2))

    # Step 4: Find h_kernel such that sum(kernel_dist) ~ 2*log(N)
    def target_function(h_kernel):
        return abs(np.sum(kernel_dist(h_kernel)) - 2 * np.log(N))

    result = minimize_scalar(target_function, bounds=(1e-5, 1e5), method="bounded")
    h_kernel = result.x
    opt_kernel_dist = kernel_dist(h_kernel)

    # Step 5: Setup the QP problem
    # Objective: min_w 0.5 * w^T H w + c^T w
    W = np.diag(weights_lsqlin)
    H = phi @ W @ phi.T
    H = 0.5 * (H + H.T)  # ensure symmetric
    # Add a small epsilon to make H positive definite if needed
    epsilon = 1e-12
    H += epsilon * np.eye(N)
    
    c = -(phi @ W @ Y)

    # Constraints:
    # sum(w) = 1  --> A w = b, where A = [1,1,...,1], b = [1]
    A = np.ones((1, N))
    b = np.array([1.0])

    # Bounds:
    # -opt_kernel_dist <= w <= opt_kernel_dist
    lb = -opt_kernel_dist
    ub = opt_kernel_dist

    # No inequality constraints Gx ≤ h needed other than bounds.
    # Solve the QP:
    # Form is:
    # min (1/2)*w^T H w + c^T w
    # s.t. A w = b and lb ≤ w ≤ ub
    # qpsolvers solve_qp interface:
    # solve_qp(P, q, G=None, h=None, A=None, b=None, lb=None, ub=None, solver=None)
    weights = solve_qp(P=H, q=c, A=A, b=b, lb=lb, ub=ub, solver='quadprog')

    return weights


if __name__ == "__main__":
    phi = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
    Y = np.array([3, 5, 7])
    weights_lsqlin = np.array([1, 0.5, 0.2])

    weights = VRx_weights(phi, Y, weights_lsqlin)
    print("Computed weights:", weights)
