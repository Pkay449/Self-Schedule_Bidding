import numpy as np
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar


def VRx_weights(phi, Y, weights_lsqlin):
    """
    Compute the weights to approximate a state.

    Parameters:
    - phi: 2D array (N x M), each row represents a state, each column a feature.
    - Y: 1D array (1 x M), the target state.
    - weights_lsqlin: 1D array (M), feature scaling weights.

    Returns:
    - weights: 1D array (N), optimized weights.
    """
    
    # Step 1: Ignore constant features
    weights_lsqlin[np.std(phi, axis=0) < 0.1] = 0

    # Step 2: Compute the weighted Euclidean distance between the states and
    # the target state
    dist = np.linalg.norm(
        (phi - Y[np.newaxis, :]) * weights_lsqlin[np.newaxis, :], axis=1
    )

    # Step 3: Compute the kernel distance
    N = len(phi)  # number of states

    def kernel_dist(h_kernel):
        return np.exp(-((dist / h_kernel) ** 2))

    # Step 4: Solve for h_kernel such that sum(kernel_dist(h_kernel)) = 2 * log(N)
    def target_function(h_kernel):
        return np.abs(np.sum(kernel_dist(h_kernel)) - 2 * np.log(N))

    result = minimize_scalar(target_function, bounds=(1e-5, 1e5), method="bounded")
    h_kernel = result.x
    opt_kernel_dist = kernel_dist(h_kernel)

    # Step 5
    # Solve the optimization problem

    lb = -opt_kernel_dist
    ub = opt_kernel_dist

    H = phi @ np.diag(weights_lsqlin) @ phi.T
    c = -phi @ np.diag(weights_lsqlin) @ Y
    
    H = (H + H.T) / 2  # Ensure H is symmetric


    # min_w 0.5 * w^T H w + c^T w
    # s.t. sum(w) = 1
    #     -kernel_dist <= w <= kernel_dist

    def objective(w):
        return 0.5 * w @ H @ w + c @ w

    def constraint(w):
        return np.sum(w) - 1

    cons = {"type": "eq", "fun": constraint}

    result = minimize(
        objective, np.zeros(N), constraints=cons, bounds=list(zip(lb, ub))
    )
    weights = result.x

    return weights


if __name__ == "__main__":
    phi = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    Y = np.array([3, 5, 7])
    weights_lsqlin = np.array([1, 0.5, 0.2])

    weights = VRx_weights(phi, Y, weights_lsqlin)

    print("Computed weights:", weights)
