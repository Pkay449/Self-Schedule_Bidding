import numpy as np
from scipy.optimize import minimize_scalar
from qpsolvers import solve_qp, available_solvers

def VRx_weights(phi, Y, weights_lsqlin):
    """
    Compute VRx weights using quadratic programming and kernel-based distance adjustment.

    Parameters:
    phi (ndarray): Feature matrix of shape (N, d), where N is the number of samples and d is the number of features.
    Y (ndarray): Target values of shape (N,) or (N, 1).
    weights_lsqlin (ndarray): Initial weights for the features.

    Returns:
    ndarray: Optimized weights of shape (N,).
    """
    if Y.ndim > 1:
        Y = Y.flatten()

    # Step 1: Ignore constant features
    mask = np.std(phi, axis=0) < 0.1
    weights_lsqlin[mask] = 0

    # Step 2: Compute weighted Euclidean distance
    weighted_diff = (phi - Y) * weights_lsqlin
    dist = np.sqrt(np.sum(weighted_diff**2, axis=1))

    N = phi.shape[0]

    def kernel_dist(h_kernel):
        return np.exp(-((dist / h_kernel) ** 2))

    # Step 4: find h_kernel
    def target_function(h_kernel):
        return abs(np.sum(kernel_dist(h_kernel)) - 2 * np.log(N))

    result = minimize_scalar(target_function, bounds=(1e-5, 1e5), method="bounded")
    h_kernel = result.x
    opt_kernel_dist = kernel_dist(h_kernel)

    # Setup QP problem
    W = np.diag(weights_lsqlin)
    H = phi @ W @ phi.T
    H = 0.5 * (H + H.T)  # ensure symmetric

    # Increase epsilon if needed
    epsilon = 1e-8
    H += epsilon * np.eye(N)

    c = -(phi @ W @ Y)
    A = np.ones((1, N))
    b = np.array([1.0])
    lb = -opt_kernel_dist
    ub = opt_kernel_dist

    # Try quadprog first
    try:
        weights = solve_qp(P=H, q=c, A=A, b=b, lb=lb, ub=ub, solver='quadprog')
    except Exception as e:
        print("quadprog failed due to positive definiteness. Trying another solver...")

        # Check what other solvers are available
        solvers = available_solvers
        # For example, try 'osqp' if available
        if 'osqp' in solvers:
            weights = solve_qp(P=H, q=c, A=A, b=b, lb=lb, ub=ub, solver='osqp')
        else:
            raise RuntimeError("No suitable solver available or all failed.")

    return weights

if __name__ == "__main__":
    phi = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
    Y = np.array([3, 5, 7])
    weights_lsqlin = np.array([1, 0.5, 0.2])

    weights = VRx_weights(phi, Y, weights_lsqlin)
    print("Computed weights:", weights)
