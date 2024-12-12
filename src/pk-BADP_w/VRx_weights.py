import numpy as np
from qpsolvers import solve_qp

def VRx_weights(phi, Y, weights_lsqlin):
    """
    Compute the weights to approximate a state.

    Parameters
    ----------
    phi : np.ndarray
        An N x D matrix of state samples.
    Y : np.ndarray
        A 1 x D vector representing the current state.
    weights_lsqlin : np.ndarray
        A D-dimensional vector of weights.

    Returns
    -------
    weights : np.ndarray
        An N-dimensional vector of weights.
    """

    # Ensure shapes are consistent
    # phi: N x D
    # Y: D or 1 x D
    # weights_lsqlin: D
    if Y.ndim > 1:
        Y = Y.flatten()
    N, D = phi.shape
    assert Y.shape[0] == D, "Y must be of length D"
    assert weights_lsqlin.shape[0] == D, "weights_lsqlin must be of length D"

    # Ignore features with std < 0.1 by setting corresponding weights to zero
    col_std = phi.std(axis=0)
    weights_lsqlin[col_std < 0.1] = 0

    # Distance computation
    # dist(n) = norm((phi(n,:) - Y) .* weights_lsqlin)
    diff = (phi - Y) * weights_lsqlin
    dist = np.sqrt(np.sum(diff**2, axis=1))

    dist_max = np.min(dist)
    h_kernel = dist_max / np.sqrt(-np.log(0.5))

    # Compute kernel distances and adjust h_kernel until sum(kernel_dist) >= 2*log(N)
    kernel_dist = np.exp(-(dist / h_kernel)**2)
    while np.sum(kernel_dist) < 2 * np.log(N):
        h_kernel = h_kernel + 1
        kernel_dist = np.exp(-(dist / h_kernel)**2)

    lb = -kernel_dist
    ub = kernel_dist

    # Construct H and f for the QP
    # H = phi * diag(weights_lsqlin) * phi'
    # shape: phi: N x D, diag: D x D, phi': D x N => H: N x N
    W = np.diag(weights_lsqlin)
    H = phi @ W @ phi.T
    # Symmetrize H
    H = 0.5 * (H + H.T)

    # # After computing H
    # H = phi @ W @ phi.T
    # H = 0.5 * (H + H.T)
    # Add a small regularization to ensure positive definiteness
    epsilon = 1e-8
    H += epsilon * np.eye(N)

    # f = -phi * diag(weights_lsqlin) * Y'
    # Y': D x 1, so result is N x 1
    f = -(phi @ W @ Y.reshape(-1,1)).flatten()

    # Inequality constraints:
    # A x ≤ b with A=[1; -1], b=[1+eps; -1+eps], eps=0
    # Actually, MATLAB code:
    # A=[ones(1,N); -ones(1,N)]
    # b=[1+eps; -1+eps]
    # With eps=0, A is 2xN, b is 2x1.
    A = np.vstack([np.ones((1,N)), -np.ones((1,N))])
    b = np.array([1, -1])

    # qpsolvers expects Gx <= h, so G=A and h=b here
    G = A
    h = b

    # Bounds: lb ≤ x ≤ ub
    # qpsolvers directly accepts lb and ub
    # QP form: minimize (1/2) x^T H x + f^T x
    # subject to G x ≤ h and lb ≤ x ≤ ub
    # x = solve_qp(P=H, q=f, G=G, h=h, A=None, b=None, lb=lb, ub=ub, solver="quadprog")

    try:
        x = solve_qp(P=H, q=f, G=G, h=h, A=None, b=None, lb=lb, ub=ub, solver="quadprog")
    except Exception as e:
        print("QP solver failed:", e)
        return None
    # 'x' is the solution which corresponds to 'weights' in MATLAB
    weights = x

    return weights
