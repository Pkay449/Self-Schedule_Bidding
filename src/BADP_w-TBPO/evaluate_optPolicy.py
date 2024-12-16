# %%
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import multivariate_normal
from scipy.spatial import ConvexHull
import os
import pickle as pkl
import warnings
import matlab.engine

# %%

# Local imports
from sample_price_day import sample_price_day
from sample_price_intraday import sample_price_intraday
from VRx_weights_pk import VRx_weights
from badp_weights_r import badp_weights

# Helper Functions
from helper import generate_scenarios, compute_weights, build_and_solve_intlinprog

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load Vt.npy, P_day_state.npy, P_intra_state.npy
Vt = np.load("Results/Vt.npy")
P_day_state = np.load("Results/P_day_state.npy")
P_intra_state = np.load("Results/P_intra_state.npy")

# =====================
# Parameters
# =====================
if False:  # Example if no arguments are given, just hardcode as in MATLAB
    length_R = 5
    N = 50
    T = 3
    M = 10
    seed = 2
    Season = "Summer"

N = 50
M = 10
T = 3
Season = "Summer"
length_R = 5
seed = 2
D = 7  # days in forecast
Rmax = 100
np.random.seed(seed)

t_ramp_pump_up = 2 / 60
t_ramp_pump_down = 2 / 60
t_ramp_turbine_up = 2 / 60
t_ramp_turbine_down = 2 / 60

c_grid_fee = 5 / 4
Delta_ti = 0.25
Delta_td = 1.0

Q_mult = 1.2
Q_fix = 3
Q_start_pump = 15
Q_start_turbine = 15

beta_pump = 0.9
beta_turbine = 0.9

x_max_pump = 10
x_min_pump = 5
x_max_turbine = 10
x_min_turbine = 5

R_vec = np.linspace(0, Rmax, length_R)
x_vec = np.array([-x_max_turbine, 0, x_max_pump])

c_pump_up = t_ramp_pump_up / 2
c_pump_down = t_ramp_pump_down / 2
c_turbine_up = t_ramp_turbine_up / 2
c_turbine_down = t_ramp_turbine_down / 2

# =====================
# Load data
# =====================
P_day_mat = loadmat(os.path.join("Data", f"P_day_{Season}.mat"))
P_intraday_mat = loadmat(os.path.join("Data", f"P_intraday_{Season}.mat"))

P_day_0 = P_day_mat["P_day_0"].flatten()
P_intraday_0 = P_intraday_mat["P_intraday_0"].flatten()

# Start MATLAB engine
eng = matlab.engine.start_matlab()


def evaluate_policy():
    """Also collect offline dataset"""

    # weights_D_value_mat = eng.badp_weights(T)
    # weights_D_value = np.array(weights_D_value_mat)

    # Collect offline dataset [s , a, r, s']
    # s = [x_start, R_start, P_day, P_intraday] 2 + 7*24 + 7*96
    # a = [day_opt, xhq_opt] 24 + 96
    # r = C
    # s' = [x_end, R_end, P_day_next, P_intraday_next] 2 + 7*24 + 7*96

    DA_s_offline = []
    DA_a_offline = []
    DA_r_offline = []
    DA_s_prime_offline = []
    
    ID_s_offline = []
    ID_a_offline = []
    ID_r_offline = []
    ID_s_prime_offline = []

    weights_D_value = badp_weights(T)

    intlinprog_options = eng.optimoptions("intlinprog", "display", "off")

    np.random.seed(seed + 1)
    (
        sample_P_day_all_fwd,
        sample_P_intraday_all_fwd,
        Wt_day_mat_fwd,
        Wt_intra_mat_fwd,
    ) = generate_scenarios(M, T, D, P_day_0, P_intraday_0, Season, seed=seed + 1)
    

    R_0 = 0
    x0_0 = 0
    V = np.zeros((M, 1))

    R_path = np.zeros((M, 96 * T))
    x_intraday_path = np.zeros((M, 96 * T))
    P_day_path = np.zeros((M, 96 * T))
    P_intraday_path = np.zeros((M, 96 * T))
    x_pump_path = np.zeros((M, 96 * T))
    x_turbine_path = np.zeros((M, 96 * T))
    y_pump_path = np.zeros((M, 96 * T))
    y_turbine_path = np.zeros((M, 96 * T))
    z_pump_path = np.zeros((M, 96 * T))
    z_turbine_path = np.zeros((M, 96 * T))

    for m in range(M):
        R = R_0
        x0 = x0_0
        P_day = P_day_0[: 24 * D].copy()
        P_intraday = P_intraday_0[: 96 * D].copy()

        P_day_sim = P_day_0[: 24 * D].copy()
        P_intraday_sim = P_intraday_0[: 96 * D].copy()

        C = 0
        for t_i in range(T):
            mu_day, _ = sample_price_day(P_day_sim, t_i, Season)
            mu_intraday, _ = sample_price_intraday(
                np.concatenate([mu_day, P_day_sim]), P_intraday_sim, t_i, Season
            )

            P_day_next = np.concatenate([mu_day, P_day[:-24].copy()])
            P_intraday_next = np.concatenate([mu_intraday, P_intraday[:-96].copy()])

            lk = 2
            VR_abc_neg = np.zeros((lk - 1, 3))
            VR_abc_pos = np.zeros((lk - 1, 3))

            # If we have another stage ahead and Vt is not empty, we need to compute weights and slopes again
            if t_i < T - 1 and np.any(Vt != 0):
                # Extract scenarios for next stage
                P_day_sample_next = P_day_state[:, t_i + 1, :].reshape(N, D * 24)
                P_intraday_sample_next = P_intra_state[:, t_i + 1, :].reshape(
                    N, D * 24 * 4
                )

                phi = np.hstack((P_day_sample_next, P_intraday_sample_next))
                Y = np.hstack((P_day_next, P_intraday_next))

                # Compute weights for next stage
                # -weights = compute_weights(eng, phi, Y, weights_D_value[int(t_i + 1), :])
                weights = VRx_weights(phi, Y, weights_D_value[int(t_i + 1), :])

                VRx = np.zeros((length_R, 3))
                for i in range(length_R):
                    for j in range(3):
                        VRx[i, j] = Vt[i, j, :, t_i + 1].dot(weights)

                hull_input = np.column_stack([R_vec.T, VRx[:, 1]])
                hull = ConvexHull(hull_input)
                k = hull.vertices
                # In MATLAB: k(1)=[] means remove the first element of k
                # Sort k and then remove the first element
                k = np.sort(k)
                k = k[1:]
                lk = len(k)

                VR = VRx[k, :]
                R_k = R_vec[k]

                if lk > 1:
                    VR_abc_neg = np.zeros((lk - 1, 3))
                    VR_abc_pos = np.zeros((lk - 1, 3))
                    for i in range(1, lk):
                        VR_abc_neg[i - 1, 1] = (VR[i, 1] - VR[i - 1, 1]) / (
                            R_k[i] - R_k[i - 1]
                        )
                        VR_abc_neg[i - 1, 0] = VR[i, 1] - VR_abc_neg[i - 1, 1] * R_k[i]
                        VR_abc_neg[i - 1, 2] = -(VR[i - 1, 1] - VR[i - 1, 0]) / (
                            x_vec[1] - x_vec[0]
                        )

                    for i in range(1, lk):
                        VR_abc_pos[i - 1, 1] = (VR[i, 1] - VR[i - 1, 1]) / (
                            R_k[i] - R_k[i - 1]
                        )
                        VR_abc_pos[i - 1, 0] = VR[i, 1] - VR_abc_pos[i - 1, 1] * R_k[i]
                        VR_abc_pos[i - 1, 2] = (VR[i - 1, 1] - VR[i - 1, 2]) / (
                            x_vec[1] - x_vec[2]
                        )
                else:
                    VR_abc_neg = np.zeros((0, 3))
                    VR_abc_pos = np.zeros((0, 3))

            # Now build the MILP for the forward pass similar to MATLAB
            # First MILP to get xday_opt
            # Build f
            f = np.zeros(96 * 12 + 24 + 1)
            f[-1] = 1
            f[96:192] -= Delta_ti * mu_intraday
            f[-25:-1] = -Delta_td * mu_day

            q_pump_up = (abs(mu_intraday) / Q_mult - Q_fix) * t_ramp_pump_up / 2
            q_pump_down = (abs(mu_intraday) * Q_mult + Q_fix) * t_ramp_pump_down / 2
            q_turbine_up = (abs(mu_intraday) * Q_mult + Q_fix) * t_ramp_turbine_up / 2
            q_turbine_down = (
                (abs(mu_intraday) / Q_mult - Q_fix) * t_ramp_turbine_down / 2
            )

            f[96 * 2 : 96 * 3] -= c_grid_fee
            f[96 * 4 : 96 * 5] += q_pump_up
            f[96 * 5 : 96 * 6] -= q_pump_down
            f[96 * 6 : 96 * 7] -= q_turbine_up
            f[96 * 7 : 96 * 8] += q_turbine_down
            f[96 * 10 : 96 * 11] -= Q_start_pump
            f[96 * 11 : 96 * 12] -= Q_start_turbine

            # Build constraints for first MILP (day-ahead)
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
            b1[0] = -R

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

            Aeq = np.vstack([A1, A2, A3, A4])
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
            b2 = np.zeros(96)
            b2[0] = float(x0 > 0)

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
            b4 = np.ones(96)

            # Since t_i<T and any(Vt(:)~=0) was checked before building slopes,
            # we already have VR_abc_neg and VR_abc_pos computed
            AV_neg = np.zeros((lk - 1, 12 * 96 + 24 + 1))
            AV_neg[:, -1] = 1
            if lk > 1:
                AV_neg[:, 96] = -VR_abc_neg[:, 1].copy()
                AV_neg[:, 4 * 96] = -VR_abc_neg[:, 2].copy()
            bV_neg = VR_abc_neg[:, 0].copy() if lk > 1 else np.array([])

            AV_pos = np.zeros((lk - 1, 12 * 96 + 24 + 1))
            AV_pos[:, -1] = 1
            if lk > 1:
                AV_pos[:, 96] = -VR_abc_neg[:, 1].copy()
                AV_pos[:, 3 * 96] = -VR_abc_neg[:, 2].copy()
            bV_pos = VR_abc_pos[:, 0].copy() if lk > 1 else np.array([])

            A = np.vstack([A1, A2, A3, A4, AV_neg, AV_pos])
            b = (
                np.concatenate([b1, b2, b3, b4, bV_neg, bV_pos])
                if lk > 1
                else np.concatenate([b1, b2, b3, b4])
            )

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

            intcon = np.arange(8 * 96, 96 * 10)
            # Solve first MILP for day-ahead
            x_opt, fval = build_and_solve_intlinprog(
                eng, f, A, b, Aeq, beq, lb, ub, intcon, intlinprog_options
            )

            xday_opt = x_opt[-25:-1].copy()

            Wt_day = Wt_day_mat_fwd[m, t_i * 24 : (t_i + 1) * 24].copy()
            day_path = np.tile(Wt_day, (4, 1))
            P_day_path[m, t_i * 96 : (t_i + 1) * 96] = day_path.flatten()

            mu_intraday, _ = sample_price_intraday(
                np.concatenate([Wt_day, P_day_sim]), P_intraday_sim, t_i, Season
            )

            P_day_next = np.concatenate([Wt_day, P_day[:-24].copy()])
            P_intraday_next = np.concatenate([mu_intraday, P_intraday[:-96].copy()])
            
            
            
            
            
            
            

            # Now solve the second MILP (intraday) with xday_opt as bounds
            # Build f again for intraday stage
            f = np.zeros(96 * 12 + 24 + 1)
            f[-1] = 1
            f[96:192] -= Delta_ti * mu_intraday
            f[-25:-1] = -Delta_td * mu_day
            q_pump_up = (abs(mu_intraday) / Q_mult - Q_fix) * t_ramp_pump_up / 2
            q_pump_down = (abs(mu_intraday) * Q_mult + Q_fix) * t_ramp_pump_down / 2
            q_turbine_up = (abs(mu_intraday) * Q_mult + Q_fix) * t_ramp_turbine_up / 2
            q_turbine_down = (
                (abs(mu_intraday) / Q_mult - Q_fix) * t_ramp_turbine_down / 2
            )
            f[96 * 2 : 96 * 3] -= c_grid_fee
            f[96 * 4 : 96 * 5] += q_pump_up
            f[96 * 5 : 96 * 6] -= q_pump_down
            f[96 * 6 : 96 * 7] -= q_turbine_up
            f[96 * 7 : 96 * 8] += q_turbine_down
            f[96 * 10 : 96 * 11] -= Q_start_pump
            f[96 * 11 : 96 * 12] -= Q_start_turbine

            # R_hq
            A1 = [
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
            A1 = np.hstack(A1)
            b1 = np.zeros(96)
            b1[0] = -R

            Axh = np.zeros((96, 24))
            for h in range(24):
                Axh[h * 4 : (h + 1) * 4, h] = -1

            A2 = [
                np.zeros((96, 96)),
                -np.eye(96),
                np.eye(96),
                -np.eye(96),
                np.zeros((96, 96 * 8)),
                Axh,
                np.zeros((96, 1)),
            ]
            A2 = np.hstack(A2)
            b2 = np.zeros(96)

            A3 = [
                np.zeros((96, 96 * 2)),
                np.eye(96) - np.diag(np.ones(95), -1),
                np.zeros((96, 96)),
                -np.eye(96),
                np.eye(96),
                np.zeros((96, 96 * 6 + 24)),
                np.zeros((96, 1)),
            ]
            A3 = np.hstack(A3)
            b3 = np.zeros(96)
            b3[0] = max(x0, 0)

            A4 = [
                np.zeros((96, 96 * 3)),
                np.eye(96) - np.diag(np.ones(95), -1),
                np.zeros((96, 96 * 2)),
                -np.eye(96),
                np.eye(96),
                np.zeros((96, 96 * 4 + 24)),
                np.zeros((96, 1)),
            ]
            A4 = np.hstack(A4)
            b4 = np.zeros(96)
            b4[0] = max(-x0, 0)

            Aeq = np.vstack([A1, A2, A3, A4])
            beq = np.hstack([b1, b2, b3, b4])

            A1 = [
                np.zeros((96, 96 * 2)),
                -np.eye(96),
                np.zeros((96, 96 * 5)),
                x_min_pump * np.eye(96),
                np.zeros((96, 96 * 3 + 24)),
                np.zeros((96, 1)),
                np.zeros(
                    (
                        96,
                        96,
                    )
                ),
            ]  # carefully build same as MATLAB

            lb = np.concatenate(
                [
                    np.zeros(96),
                    -np.inf * np.ones(96),
                    np.zeros(96 * 10),
                    xday_opt,
                    np.full(1, -np.inf),
                ]
            )
            ub = np.concatenate(
                [
                    Rmax * np.ones(96),
                    np.inf * np.ones(96 * 7),
                    np.ones(96 * 4),
                    xday_opt,
                    np.full(1, np.inf),
                ]
            )

            x_opt2, fval2 = build_and_solve_intlinprog(
                eng, f, A, b, Aeq, beq, lb, ub, intcon, intlinprog_options
            )

            # Extract results from x_opt2
            R_opt = x_opt2[:96].copy()
            xhq_opt = x_opt2[96 : 2 * 96].copy()

            Delta_pump_up = x_opt2[4 * 96 : 5 * 96].copy()
            Delta_pump_down = x_opt2[5 * 96 : 6 * 96].copy()
            Delta_turbine_up = x_opt2[6 * 96 : 7 * 96].copy()
            Delta_turbine_down = x_opt2[7 * 96 : 8 * 96].copy()

            x_pump = x_opt2[2 * 96 : 3 * 96].copy()
            x_turbine = x_opt2[3 * 96 : 4 * 96].copy()
            y_pump = x_opt2[8 * 96 : 9 * 96].copy()
            y_turbine = x_opt2[9 * 96 : 10 * 96].copy()
            z_pump = x_opt2[10 * 96 : 11 * 96].copy()
            z_turbine = x_opt2[11 * 96 : 12 * 96].copy()

            # Update paths
            R_path[m, t_i * 96 : (t_i + 1) * 96] = R_opt
            x_intraday_path[m, t_i * 96 : (t_i + 1) * 96] = xhq_opt
            x_pump_path[m, t_i * 96 : (t_i + 1) * 96] = x_pump
            x_turbine_path[m, t_i * 96 : (t_i + 1) * 96] = x_turbine
            y_pump_path[m, t_i * 96 : (t_i + 1) * 96] = y_pump
            y_turbine_path[m, t_i * 96 : (t_i + 1) * 96] = y_turbine
            z_pump_path[m, t_i * 96 : (t_i + 1) * 96] = z_pump
            z_turbine_path[m, t_i * 96 : (t_i + 1) * 96] = z_turbine

            Wt_intraday = Wt_intra_mat_fwd[m, t_i * 96 : (t_i + 1) * 96].copy()
            P_intraday_path[m, t_i * 96 : (t_i + 1) * 96] = Wt_intraday

            # Update q_ ramps with realized intraday prices
            q_pump_up = (np.abs(Wt_intraday) / Q_mult - Q_fix) * t_ramp_pump_up / 2
            q_pump_down = (np.abs(Wt_intraday) * Q_mult + Q_fix) * t_ramp_pump_down / 2
            q_turbine_up = (
                (np.abs(Wt_intraday) * Q_mult + Q_fix) * t_ramp_turbine_up / 2
            )
            q_turbine_down = (
                (np.abs(Wt_intraday) / Q_mult - Q_fix) * t_ramp_turbine_down / 2
            )

            state = np.concatenate([[x0], [R], P_day, P_intraday])
            action = np.concatenate([xday_opt, xhq_opt])

            # Update R, x0, P_day, P_intraday, P_day_sim, P_intraday_sim
            R = R_opt[-1].copy()
            x0 = x_pump[-1] - x_turbine[-1]
            P_day = np.concatenate([Wt_day, P_day[:-24].copy()])
            P_intraday = np.concatenate([Wt_intraday, P_intraday[:-96].copy()])
            P_day_sim = np.concatenate([Wt_day, P_day_sim[:-24].copy()])
            P_intraday_sim = np.concatenate([Wt_intraday, P_intraday_sim[:-96].copy()])

            next_state = np.concatenate([[x0], [R], P_day, P_intraday])

            # Update C
            reward = (-Delta_td * np.dot(Wt_day, xday_opt)
                                 - np.sum(x_pump) * c_grid_fee
                                 - Delta_ti * np.dot(Wt_intraday, xhq_opt)
                                 + np.dot(q_pump_up, Delta_pump_up)
                                 - np.dot(q_pump_down, Delta_pump_down)
                                 - np.dot(q_turbine_up, Delta_turbine_up)
                                 + np.dot(q_turbine_down, Delta_turbine_down)
                                 - np.sum(z_pump) * Q_start_pump
                                 - np.sum(z_turbine) * Q_start_turbine)
            C += reward

            s_offline.append(state)
            a_offline.append(action)
            r_offline.append(reward)
            s_prime_offline.append(next_state)

        V[m] = C

    EV = np.mean(V)
    print(EV)
    
    # Save offline dataset as pickle
    df = pd.DataFrame({
        'state': s_offline,
        'action': a_offline,
        'reward': r_offline,
        'next_state': s_prime_offline
    })
    df.to_pickle('Results/offline_dataset.pkl')


    return EV


if __name__ == "__main__":
    evaluate_policy()
