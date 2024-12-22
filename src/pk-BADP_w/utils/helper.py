import numpy as np
from scipy.io import loadmat
from scipy.stats import multivariate_normal
from scipy.spatial import ConvexHull
import os
import warnings
import matlab.engine

# Local imports
from .sample_price_day import sample_price_day
from .sample_price_intraday import sample_price_intraday
from .VRx_weights_pk import VRx_weights
from .badp_weights_r import badp_weights

# =====================
# Helper Functions
# =====================


def generate_scenarios(N, T, D, P_day_0, P_intraday_0, Season, seed=None):
    if seed is not None:
        np.random.seed(seed)
    sample_P_day_all = np.zeros((N, T, 24 * D))
    sample_P_intraday_all = np.zeros((N, T, 96 * D))
    Wt_day_mat = np.zeros((N, T * 24))
    Wt_intra_mat = np.zeros((N, T * 96))
    for n in range(N):
        P_day = P_day_0[: 24 * D].copy()
        P_intraday = P_intraday_0[: 96 * D].copy()
        for t_i in range(T):
            sample_P_day_all[n, t_i, :] = P_day
            sample_P_intraday_all[n, t_i, :] = P_intraday
            mu_day, cor_day = sample_price_day(P_day, t_i, Season)
            Wt_day = multivariate_normal.rvs(mean=mu_day, cov=cor_day)

            mu_intraday, cor_intraday = sample_price_intraday(
                np.concatenate([Wt_day, P_day]), P_intraday, t_i, Season
            )
            Wt_intraday = multivariate_normal.rvs(mean=mu_intraday, cov=cor_intraday)

            P_day = np.concatenate([Wt_day, P_day[:-24]])
            P_intraday = np.concatenate([Wt_intraday, P_intraday[:-96]])

            Wt_day_mat[n, t_i * 24 : (t_i + 1) * 24] = Wt_day
            Wt_intra_mat[n, t_i * 96 : (t_i + 1) * 96] = Wt_intraday
    return sample_P_day_all, sample_P_intraday_all, Wt_day_mat, Wt_intra_mat


def compute_weights(eng, phi, Y, weights_vector):
    weights = VRx_weights(phi, Y, weights_vector)
    return weights


def build_and_solve_intlinprog(eng, f, A, b, Aeq, beq, lb, ub, intcon, options):
    matlab_f = matlab.double((-f).tolist())  # MATLAB: minimize
    matlab_A = matlab.double(A.tolist())
    matlab_b = matlab.double(b.tolist())
    matlab_Aeq = matlab.double(Aeq.tolist())
    matlab_beq = matlab.double(beq.tolist())
    matlab_lb = matlab.double(lb.tolist())
    matlab_ub = matlab.double(ub.tolist())
    matlab_intcon = matlab.double((intcon + 1).tolist())
    xres, fvalres = eng.intlinprog(
        matlab_f,
        matlab_intcon,
        matlab_A,
        matlab_b,
        matlab_Aeq,
        matlab_beq,
        matlab_lb,
        matlab_ub,
        options,
        nargout=2,
    )
    x_opt = np.array(xres).flatten()
    fval = float(fvalres)
    return x_opt, fval


def linear_constraints_train(
    Delta_ti,
    beta_pump,
    beta_turbine,
    c_pump_up,
    c_pump_down,
    c_turbine_up,
    c_turbine_down,
    R_val,
    x0,
    x_min_pump,
    x_max_pump,
    x_min_turbine,
    x_max_turbine,
    Rmax,
    lk,
    VR_abc_neg,
    VR_abc_pos,
):
    # Linear constraints
    # A1
    A1 = np.hstack(
        [
            -np.eye(96) + np.diag(np.ones(95), -1),
            np.zeros((96, 96)),
            Delta_ti * beta_pump * np.eye(96),
            -Delta_ti / beta_turbine * np.eye(96),
            -beta_pump * c_pump_up * np.eye(96),
            beta_pump * c_pump_down * np.eye(96),
            c_turbine_up / beta_turbine * np.eye(96),
            -c_turbine_down / beta_turbine * np.eye(96),
            np.zeros((96, 96 * 4 + 24)),
            np.zeros((96, 1)),
        ]
    )
    b1 = np.zeros(96)
    b1[0] = -R_val

    # A2
    Axh = np.zeros((96, 24))
    for h in range(24):
        Axh[h * 4 : (h + 1) * 4, h] = -1

    A2 = np.hstack(
        [
            np.zeros((96, 96)),
            -np.eye(96),
            np.eye(96),
            -np.eye(96),
            np.zeros((96, 96 * 8)),
            Axh,
            np.zeros((96, 1)),
        ]
    )
    b2 = np.zeros(96)

    A3 = np.hstack(
        [
            np.zeros((96, 96 * 2)),
            np.eye(96) - np.diag(np.ones(95), -1),
            np.zeros((96, 96)),
            -np.eye(96),
            np.eye(96),
            np.zeros((96, 96 * 6 + 24)),
            np.zeros((96, 1)),
        ]
    )
    b3 = np.zeros(96)
    b3[0] = max(x0, 0)

    A4 = np.hstack(
        [
            np.zeros((96, 96 * 3)),
            np.eye(96) - np.diag(np.ones(95), -1),
            np.zeros((96, 96 * 2)),
            -np.eye(96),
            np.eye(96),
            np.zeros((96, 96 * 4 + 24)),
            np.zeros((96, 1)),
        ]
    )
    b4 = np.zeros(96)
    b4[0] = max(-x0, 0)

    # Stack A1, A2, A3, A4 vertically (row-wise)
    Aeq = np.vstack([A1, A2, A3, A4])

    # Stack b1, b2, b3, b4 vertically (row-wise)
    beq = np.hstack([b1, b2, b3, b4])

    A1 = np.vstack(
        [
            np.hstack(
                [
                    np.zeros((96, 96 * 2)),
                    -np.eye(96),
                    np.zeros((96, 96 * 5)),
                    x_min_pump * np.eye(96),
                    np.zeros((96, 96 * 3 + 24)),
                    np.zeros((96, 1)),
                ]
            ),
            np.hstack(
                [
                    np.zeros((96, 96 * 2)),
                    np.eye(96),
                    np.zeros((96, 96 * 5)),
                    -x_max_pump * np.eye(96),
                    np.zeros((96, 96 * 3 + 24)),
                    np.zeros((96, 1)),
                ]
            ),
            np.hstack(
                [
                    np.zeros((96, 96 * 3)),
                    -np.eye(96),
                    np.zeros((96, 96 * 5)),
                    x_min_turbine * np.eye(96),
                    np.zeros((96, 96 * 2 + 24)),
                    np.zeros((96, 1)),
                ]
            ),
            np.hstack(
                [
                    np.zeros((96, 96 * 3)),
                    np.eye(96),
                    np.zeros((96, 96 * 5)),
                    -x_max_turbine * np.eye(96),
                    np.zeros((96, 96 * 2 + 24)),
                    np.zeros((96, 1)),
                ]
            ),
        ]
    )

    b1 = np.zeros(96 * 4)

    A2 = np.hstack(
        [
            np.zeros((96, 96 * 8)),
            np.eye(96) - np.diag(np.ones(95), -1),
            np.zeros((96, 96)),
            -np.eye(96),
            np.zeros((96, 96 + 24)),
            np.zeros((96, 1)),
        ]
    )

    # Construct b2
    b2 = np.zeros(96)
    b2[0] = float(x0 > 0)

    # Construct A3
    A3 = np.hstack(
        [
            np.zeros((96, 96 * 9)),
            np.eye(96) - np.diag(np.ones(95), -1),
            np.zeros((96, 96)),
            -np.eye(96),
            np.zeros((96, 24)),
            np.zeros((96, 1)),
        ]
    )

    # Construct b3
    b3 = np.zeros(96)
    b3[0] = float(x0 < 0)

    A4 = np.hstack(
        [
            np.zeros((96, 96 * 8)),
            np.eye(96),
            np.eye(96),
            np.zeros((96, 2 * 96 + 24)),
            np.zeros((96, 1)),
        ]
    )

    # Construct b4
    b4 = np.ones(96)

    AV_neg = np.zeros((lk - 1, 12 * 96 + 24 + 1))
    AV_neg[:, -1] = 1
    AV_neg[:, 96] = -VR_abc_neg[:, 1].copy()
    AV_neg[:, 4 * 96] = -VR_abc_neg[:, 2].copy()
    bV_neg = VR_abc_neg[:, 0].copy()

    AV_pos = np.zeros((lk - 1, 12 * 96 + 24 + 1))
    AV_pos[:, -1] = 1
    AV_pos[:, 96] = -VR_abc_neg[:, 1].copy()
    AV_pos[:, 3 * 96] = -VR_abc_neg[:, 2].copy()
    bV_pos = VR_abc_pos[:, 0]

    A = np.vstack([A1, A2, A3, A4, AV_neg, AV_pos])
    b = np.concatenate([b1, b2, b3, b4, bV_neg, bV_pos])

    lb = np.concatenate(
        [
            np.zeros(96),
            -np.inf * np.ones(96),
            np.zeros(96 * 10),
            -x_max_turbine * np.ones(24),
            np.full(1, -np.inf),
        ]
    )
    ub = np.concatenate(
        [
            Rmax * np.ones(96),
            np.inf * np.ones(96 * 7),
            np.ones(96 * 4),
            x_max_pump * np.ones(24),
            np.full(1, np.inf),
        ]
    )

    return A, b, Aeq, beq, lb, ub


def linear_constraints_train_FVI(
    Delta_ti,
    beta_pump,
    beta_turbine,
    c_pump_up,
    c_pump_down,
    c_turbine_up,
    c_turbine_down,
    R_val,
    x0,
    x_min_pump,
    x_max_pump,
    x_min_turbine,
    x_max_turbine,
    Rmax,
):
    # Construct constraints as before, but without convex hull logic:
    # A1
    A1 = np.hstack(
        [
            -np.eye(96) + np.diag(np.ones(95), -1),
            np.zeros((96, 96)),
            Delta_ti * beta_pump * np.eye(96),
            -Delta_ti / beta_turbine * np.eye(96),
            -beta_pump * c_pump_up * np.eye(96),
            beta_pump * c_pump_down * np.eye(96),
            c_turbine_up / beta_turbine * np.eye(96),
            -c_turbine_down / beta_turbine * np.eye(96),
            np.zeros((96, 96 * 4 + 24)),
            np.zeros((96, 1)),
        ]
    )
    b1 = np.zeros(96)
    b1[0] = -R_val

    # A2
    Axh = np.zeros((96, 24))
    for h in range(24):
        Axh[h * 4:(h + 1) * 4, h] = -1

    A2 = np.hstack(
        [
            np.zeros((96, 96)),
            -np.eye(96),
            np.eye(96),
            -np.eye(96),
            np.zeros((96, 96 * 8)),
            Axh,
            np.zeros((96, 1)),
        ]
    )
    b2 = np.zeros(96)

    A3 = np.hstack(
        [
            np.zeros((96, 96*2)),
            np.eye(96) - np.diag(np.ones(95), -1),
            np.zeros((96, 96)),
            -np.eye(96),
            np.eye(96),
            np.zeros((96, 96*6 + 24)),
            np.zeros((96, 1)),
        ]
    )
    b3 = np.zeros(96)
    b3[0] = max(x0, 0)

    A4 = np.hstack(
        [
            np.zeros((96, 96*3)),
            np.eye(96) - np.diag(np.ones(95), -1),
            np.zeros((96, 96*2)),
            -np.eye(96),
            np.eye(96),
            np.zeros((96, 96*4 + 24)),
            np.zeros((96, 1)),
        ]
    )
    b4 = np.zeros(96)
    b4[0] = max(-x0, 0)

    Aeq = np.vstack([A1, A2, A3, A4])
    beq = np.hstack([b1, b2, b3, b4])

    # Constraints for pump and turbine power limits
    A1 = np.vstack(
        [
            np.hstack(
                [
                    np.zeros((96, 96*2)),
                    -np.eye(96),
                    np.zeros((96, 96*5)),
                    x_min_pump * np.eye(96),
                    np.zeros((96, 96*3+24)),
                    np.zeros((96, 1)),
                ]
            ),
            np.hstack(
                [
                    np.zeros((96, 96*2)),
                    np.eye(96),
                    np.zeros((96, 96*5)),
                    -x_max_pump * np.eye(96),
                    np.zeros((96, 96*3+24)),
                    np.zeros((96, 1)),
                ]
            ),
            np.hstack(
                [
                    np.zeros((96, 96*3)),
                    -np.eye(96),
                    np.zeros((96, 96*5)),
                    x_min_turbine * np.eye(96),
                    np.zeros((96, 96*2+24)),
                    np.zeros((96, 1)),
                ]
            ),
            np.hstack(
                [
                    np.zeros((96, 96*3)),
                    np.eye(96),
                    np.zeros((96, 96*5)),
                    -x_max_turbine * np.eye(96),
                    np.zeros((96, 96*2+24)),
                    np.zeros((96, 1)),
                ]
            ),
        ]
    )
    b1 = np.zeros(96*4)

    # Additional constraints if needed:
    A2 = np.hstack(
        [
            np.zeros((96, 96*8)),
            np.eye(96) - np.diag(np.ones(95), -1),
            np.zeros((96, 96)),
            -np.eye(96),
            np.zeros((96, 96+24)),
            np.zeros((96, 1)),
        ]
    )
    b2 = np.zeros(96)
    b2[0] = float(x0 > 0)

    A3 = np.hstack(
        [
            np.zeros((96, 96*9)),
            np.eye(96)-np.diag(np.ones(95),-1),
            np.zeros((96,96)),
            -np.eye(96),
            np.zeros((96,24)),
            np.zeros((96,1))
        ]
    )
    b3 = np.zeros(96)
    b3[0] = float(x0 < 0)

    A4 = np.hstack(
        [
            np.zeros((96,96*8)),
            np.eye(96),
            np.eye(96),
            np.zeros((96,2*96+24)),
            np.zeros((96,1))
        ]
    )
    b4 = np.ones(96)

    # Remove AV_neg and AV_pos logic entirely since no convex hull:
    # No VR_abc_neg or VR_abc_pos usage.

    A = np.vstack([A1, A2, A3, A4])
    b = np.concatenate([b1, b2, b3, b4])

    lb = np.concatenate(
        [
            np.zeros(96),
            -np.inf * np.ones(96),
            np.zeros(96*10),
            -x_max_turbine * np.ones(24),
            np.full(1, -np.inf),
        ]
    )
    ub = np.concatenate(
        [
            Rmax * np.ones(96),
            np.inf * np.ones(96*7),
            np.ones(96*4),
            x_max_pump * np.ones(24),
            np.full(1, np.inf),
        ]
    )

    return A, b, Aeq, beq, lb, ub
