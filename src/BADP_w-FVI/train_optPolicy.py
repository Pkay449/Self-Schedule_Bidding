# %%
import numpy as np
from scipy.io import loadmat
from scipy.stats import multivariate_normal
from scipy.spatial import ConvexHull
import os
import warnings
import matlab.engine

# Local imports
from sample_price_day import sample_price_day
from sample_price_intraday import sample_price_intraday
from VRx_weights_pk import VRx_weights
from badp_weights_r import badp_weights

# Helper Functions
from helper import (
    generate_scenarios,
    compute_weights,
    build_and_solve_intlinprog,
    linear_constraints_train,
)

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

# weights_D_value_mat = eng.badp_weights(T)
# weights_D_value = np.array(weights_D_value_mat)

weights_D_value = badp_weights(T)

intlinprog_options = eng.optimoptions("intlinprog", "display", "off")

# =====================
# Backward pass
# =====================
sample_P_day_all, sample_P_intraday_all, Wt_day_mat, Wt_intra_mat = generate_scenarios(
    N, T, D, P_day_0, P_intraday_0, Season, seed=seed
)

P_day_state = sample_P_day_all.copy()
P_intra_state = sample_P_intraday_all.copy()

Vt = np.zeros((length_R, 3, N, T + 1))

for t_i in range(T - 1, -1, -1):
    P_day_sample = P_day_state[:, t_i, :].copy()
    P_intraday_sample = P_intra_state[:, t_i, :].copy()

    if t_i < T - 1:
        P_day_sample_next = P_day_state[:, t_i + 1, :].reshape(N, D * 24)
        P_intraday_sample_next = P_intra_state[:, t_i + 1, :].reshape(N, D * 24 * 4)

    for n in range(N):
        P_day = P_day_sample[n, :].copy()
        P_intraday = P_intraday_sample[n, :].copy()

        mu_day, cor_day = sample_price_day(P_day, t_i, Season)
        mu_intraday, cor_intraday = sample_price_intraday(
            np.concatenate([mu_day, P_day]), P_intraday, t_i, Season
        )

        P_day_next = np.concatenate([mu_day, P_day[:-24]])
        P_intraday_next = np.concatenate([mu_intraday, P_intraday[:-96]])

        lk = 2
        VR_abc_neg = np.zeros((lk - 1, 3))
        VR_abc_pos = np.zeros((lk - 1, 3))

        if t_i < T - 1:
            phi = np.concatenate([P_day_sample_next, P_intraday_sample_next], axis=1)
            Y = np.concatenate([P_day_next, P_intraday_next])
            weights = VRx_weights(phi, Y, weights_D_value[int(t_i + 1), :])

            VRx = np.zeros((length_R, 3))
            for i in range(length_R):
                for j in range(3):
                    VRx[i, j] = Vt[i, j, :, t_i + 1].dot(weights)

            hull_input = np.column_stack([R_vec.T, VRx[:, 1]])
            hull = ConvexHull(hull_input)
            k = hull.vertices
            k = np.sort(k)[::-1]
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

        # Solve MILP for each state (R, x0)
        for iR in range(length_R):
            R_val = R_vec[iR]
            for ix in range(len(x_vec)):
                x0 = x_vec[ix]

                # Build f
                f = np.zeros(96 * 12 + 24 + 1)
                f[-1] = 1
                # As per code:
                f[96:192] -= Delta_ti * mu_intraday
                f[-25:-1] = -Delta_td * mu_day
                q_pump_up = (abs(mu_intraday) / Q_mult - Q_fix) * t_ramp_pump_up / 2
                q_pump_down = (abs(mu_intraday) * Q_mult + Q_fix) * t_ramp_pump_down / 2
                q_turbine_up = (
                    (abs(mu_intraday) * Q_mult + Q_fix) * t_ramp_turbine_up / 2
                )
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

                A, b, Aeq, beq, lb, ub = linear_constraints_train(
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
                )

                intcon = np.arange(8 * 96, 96 * 10)
                
                x_opt, fval = build_and_solve_intlinprog(eng, f, A, b, Aeq, beq, lb, ub, intcon, intlinprog_options)

                Vt[iR, ix, n, t_i] = -fval

# Save Vt, P_day_state, P_intra_state in
np.save("Results/Vt.npy", Vt)
np.save("Results/P_day_state.npy", P_day_state)
np.save("Results/P_intra_state.npy", P_intra_state)