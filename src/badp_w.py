import numpy as np
from scipy.io import loadmat
from scipy.stats import multivariate_normal
from scipy.optimize import milp, linprog, LinearConstraint, Bounds
from scipy.spatial import ConvexHull
# If milp is not available, consider using pulp or mip.

# Import local functions
from badp_weights import badp_weights
from VRx_weights import VRx_weights
from sample_price_day import sample_price_day
from sample_price_intraday import sample_price_intraday
import os

import warnings

warnings.filterwarnings("ignore")

# set the current working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

N = 50
M = 10
T = 3
Season = 'Summer'
length_R = 5
seed = 1

D = 7  # days in forecast
np.random.seed(seed)
Rmax = 100

weights_D_value = badp_weights(T)

# Parameters
t_ramp_pump_up = 2/60
t_ramp_pump_down = 2/60
t_ramp_turbine_up = 2/60
t_ramp_turbine_down = 2/60

c_grid_fee = 5/4
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

c_pump_up = t_ramp_pump_up/2
c_pump_down = t_ramp_pump_down/2
c_turbine_up = t_ramp_turbine_up/2
c_turbine_down = t_ramp_turbine_down/2

P_day_mat = loadmat(f'Data/P_day_{Season}.mat')
P_intraday_mat = loadmat(f'Data/P_intraday_{Season}.mat')

P_day_0 = P_day_mat['P_day_0'].flatten()
P_intraday_0 = P_intraday_mat['P_intraday_0'].flatten()

# Sample paths for backward calculation
sample_P_day_all = np.zeros((N, T, 24*D))
sample_P_intraday_all = np.zeros((N, T, 96*D))

Wt_day_mat = np.zeros((N, T*24))
Wt_intra_mat = np.zeros((N, T*96))

for n in range(N):
    P_day = P_day_0[:24*D].copy()
    P_intraday = P_intraday_0[:96*D].copy()
    for t_i in range(T):
        sample_P_day_all[n, t_i, :] = P_day
        sample_P_intraday_all[n, t_i, :] = P_intraday

        mu_day, cor_day = sample_price_day(P_day, t_i, Season)
        Wt_day = multivariate_normal.rvs(mean=mu_day, cov=cor_day)

        mu_intraday, cor_intraday = sample_price_intraday(np.concatenate([Wt_day, P_day]), P_intraday, t_i, Season)
        Wt_intraday = multivariate_normal.rvs(mean=mu_intraday, cov=cor_intraday)

        P_day = np.concatenate([Wt_day, P_day[:-24]])
        P_intraday = np.concatenate([Wt_intraday, P_intraday[:-96]])

        Wt_day_mat[n, t_i*24:(t_i+1)*24] = Wt_day
        Wt_intra_mat[n, t_i*96:(t_i+1)*96] = Wt_intraday

P_day_state = sample_P_day_all.copy()
P_intra_state = sample_P_intraday_all.copy()

# Initialize Vt: shape (length_R, 3, N, T+1)
Vt = np.zeros((length_R, 3, N, T+1))

# Backward step
for t_i in range(T-1, -1, -1):
    P_day_sample = P_day_state[:, t_i, :].copy()  # shape (N, 24*D)
    P_intraday_sample = P_intra_state[:, t_i, :].copy() # shape (N, 96*D)

    if t_i < T-1:
        P_day_sample_next = P_day_state[:, t_i + 1, :].copy().reshape(N, D * 24)
        P_intraday_sample_next = P_intra_state[:, t_i + 1, :].copy().reshape(N, D * 24 * 4)

    for n in range(N):
        P_day = P_day_sample[n, :].copy()
        P_intraday = P_intraday_sample[n, :].copy()

        mu_day, cor_day = sample_price_day(P_day, t_i, Season)
        mu_intraday, cor_intraday = sample_price_intraday(np.concatenate([mu_day, P_day]), P_intraday, t_i, Season)

        P_day_next = np.concatenate([mu_day, P_day[:-24]])
        P_intraday_next = np.concatenate([mu_intraday, P_intraday[:-96]])

        # Construct the piecewise linear approximations
        lk = 2
        VR_abc_neg = np.zeros((lk-1, 3))
        VR_abc_pos = np.zeros((lk-1, 3))

        if t_i < T-1:
            phi = np.concatenate([P_day_sample_next, P_intraday_sample_next], axis=1)
            Y = np.concatenate([P_day_next, P_intraday_next])
            weights = VRx_weights(phi, Y, weights_D_value[t_i+1, :].copy())

            VRx = np.zeros((length_R, 3))
            for i in range(length_R):
                for j in range(3):
                    VRx[i, j] = np.dot(Vt[i, j, :, t_i + 1].copy().reshape(1, N), weights)

            # Create input for convex hull computation
            hull_input = np.column_stack([R_vec.T, VRx[:, 1]])

            # Compute the convex hull
            hull = ConvexHull(hull_input)
            k = hull.vertices
            # print(k)

            # Sort k by R_vec values to ensure ordering consistency
            # k = k[np.argsort(R_vec[k])]

            # Remove the first element (equivalent to `k(1)=[]` in MATLAB)
            # k = k[1:]
            # print(k)
            # break
            k = np.sort(k)[::-1]

            # Length of k
            lk = len(k)

            # Initialize arrays for a, b, c values
            VR_abc_neg = np.zeros((lk - 1, 3))
            VR_abc_pos = np.zeros((lk - 1, 3))

            # Extract rows from VRx corresponding to k (equivalent to `VR = VRx(k, :)`)
            VR = VRx[k, :].copy()

            # Extract corresponding R_vec values
            R_k = R_vec[k]

            # Build VR_abc_neg, VR_abc_pos similarly as MATLAB code
            # This is left as an exercise due to complexity.
            # For now, assume VR_abc_neg, VR_abc_pos are computed similarly as in MATLAB code.
            # Loop for VR_abc_neg
            for i in range(1, lk):  # MATLAB indexing starts from 2; Python indexing starts from 1
                VR_abc_neg[i - 1, 1] = (VR[i, 1] - VR[i - 1, 1]) / (R_k[i] - R_k[i - 1])  # Steigung R
                VR_abc_neg[i - 1, 0] = VR[i, 1] - VR_abc_neg[i - 1, 1] * R_k[i]  # Achsenabschnitt
                VR_abc_neg[i - 1, 2] = -(VR[i - 1, 1] - VR[i - 1, 0]) / (x_vec[1] - x_vec[0])  # Steigung x

            # Loop for VR_abc_pos
            for i in range(1, lk):  # MATLAB indexing starts from 2; Python indexing starts from 1
                VR_abc_pos[i - 1, 1] = (VR[i, 1] - VR[i - 1, 1]) / (R_k[i] - R_k[i - 1])  # Steigung R
                VR_abc_pos[i - 1, 0] = VR[i, 1] - VR_abc_pos[i - 1, 1] * R_k[i]  # Achsenabschnitt
                VR_abc_pos[i - 1, 2] = (VR[i - 1, 1] - VR[i - 1, 2]) / (x_vec[1] - x_vec[2])  # Steigung x

        # For each R and x
        for iR in range(length_R):
            R = R_vec[iR]
            for ix in range(len(x_vec)):
                x0 = x_vec[ix]
                f = np.zeros(96 * 12 + 24 + 1)
                f[-1] = 1  # End value

                # Single-stage objective function
                f[96:96 * 2] -= Delta_ti * mu_intraday
                f[-25:-1] = -Delta_td * mu_day

                q_pump_up = (abs(mu_intraday) / Q_mult - Q_fix) * t_ramp_pump_up / 2
                q_pump_down = (abs(mu_intraday) * Q_mult + Q_fix) * t_ramp_pump_down / 2
                q_turbine_up = (abs(mu_intraday) * Q_mult + Q_fix) * t_ramp_turbine_up / 2
                q_turbine_down = (abs(mu_intraday) / Q_mult - Q_fix) * t_ramp_turbine_down / 2

                # Grid fee
                f[96 * 2:96 * 3] -= c_grid_fee
                # delta_pump_up
                f[96 * 4:96 * 5] += q_pump_up
                # delta_pump_down
                f[96 * 5:96 * 6] -= q_pump_down
                # delta_turbine_up
                f[96 * 6:96 * 7] -= q_turbine_up
                # delta_turbine_down
                f[96 * 7:96 * 8] += q_turbine_down

                # z^pump
                f[96 * 10:96 * 11] -= Q_start_pump
                # z^turbine
                f[96 * 11:96 * 12] -= Q_start_turbine

                # Linear constraints
                # A1
                A1 = np.hstack([
                    -np.eye(96) + np.diag(np.ones(95), -1),
                    np.zeros((96, 96)),
                    Delta_ti * beta_pump * np.eye(96),
                    -Delta_ti / beta_turbine * np.eye(96),
                    -beta_pump * c_pump_up * np.eye(96),
                    beta_pump * c_pump_down * np.eye(96),
                    c_turbine_up / beta_turbine * np.eye(96),
                    -c_turbine_down / beta_turbine * np.eye(96),
                    np.zeros((96, 96 * 4 + 24)),
                    np.zeros((96, 1))
                ])
                b1 = np.zeros(96)
                b1[0] = -R

                # A2
                Axh = np.zeros((96, 24))
                for h in range(24):
                    Axh[h * 4:(h + 1) * 4, h] = -1

                A2 = np.hstack([
                    np.zeros((96, 96)),
                    -np.eye(96),
                    np.eye(96),
                    -np.eye(96),
                    np.zeros((96, 96 * 8)),
                    Axh,
                    np.zeros((96, 1))
                ])
                b2 = np.zeros(96)

                A3 = np.hstack([
                    np.zeros((96, 96*2)),
                    np.eye(96) - np.diag(np.ones(95), -1),
                    np.zeros((96, 96)),
                    -np.eye(96),
                    np.eye(96),
                    np.zeros((96, 96*6 + 24)),
                    np.zeros((96, 1))
                ])
                b3 = np.zeros(96)
                b3[0] = max(x0, 0)

                A4 = np.hstack([
                    np.zeros((96, 96*3)),
                    np.eye(96) - np.diag(np.ones(95), -1),
                    np.zeros((96, 96*2)),
                    -np.eye(96),
                    np.eye(96),
                    np.zeros((96, 96*4 + 24)),
                    np.zeros((96, 1))
                ])
                b4=np.zeros(96)
                b4[0]=max(-x0,0)

                # Stack A1, A2, A3, A4 vertically (row-wise)
                Aeq = np.vstack([A1, A2, A3, A4])

                # Stack b1, b2, b3, b4 vertically (row-wise)
                beq = np.hstack([b1, b2, b3, b4])

                A1 = np.vstack([
                    np.hstack([np.zeros((96, 96*2)), -np.eye(96), np.zeros((96, 96*5)), x_min_pump * np.eye(96), np.zeros((96, 96*3+24)), np.zeros((96, 1))]),
                    np.hstack([np.zeros((96, 96*2)), np.eye(96), np.zeros((96, 96*5)), -x_max_pump * np.eye(96), np.zeros((96, 96*3+24)), np.zeros((96, 1))]),
                    np.hstack([np.zeros((96, 96*3)), -np.eye(96), np.zeros((96, 96*5)), x_min_turbine * np.eye(96), np.zeros((96, 96*2+24)), np.zeros((96, 1))]),
                    np.hstack([np.zeros((96, 96*3)), np.eye(96), np.zeros((96, 96*5)), -x_max_turbine * np.eye(96), np.zeros((96, 96*2+24)), np.zeros((96, 1))])
                ])

                b1=np.zeros(96*4)

                A2 = np.hstack([
                    np.zeros((96, 96*8)),
                    np.eye(96) - np.diag(np.ones(95), -1),
                    np.zeros((96, 96)),
                    -np.eye(96),
                    np.zeros((96, 96 + 24)),
                    np.zeros((96, 1))
                ])

                # Construct b2
                b2 = np.zeros(96)
                b2[0] = x0 > 0

                # Construct A3
                A3 = np.hstack([
                    np.zeros((96, 96*9)),
                    np.eye(96) - np.diag(np.ones(95), -1),
                    np.zeros((96, 96)),
                    -np.eye(96),
                    np.zeros((96, 24)),
                    np.zeros((96, 1))
                ])

                # Construct b3
                b3 = np.zeros(96)
                b3[0] = x0 < 0

                A4 = np.hstack([
                    np.zeros((96, 96*8)),
                    np.eye(96),
                    np.eye(96),
                    np.zeros((96, 2*96+24)),
                    np.zeros((96, 1))
                ])

                # Construct b4
                b4 = np.ones(96)

                AV_neg = np.zeros((lk-1, 12*96 + 24 + 1))
                AV_neg[:, -1] = 1
                AV_neg[:, 96] = -VR_abc_neg[:, 1].copy()
                AV_neg[:, 4*96] = -VR_abc_neg[:, 2].copy()
                bV_neg = VR_abc_neg[:, 0].copy()

                AV_pos = np.zeros((lk-1, 12*96 + 24 + 1))
                AV_pos[:, -1] = 1
                AV_pos[:, 96] = -VR_abc_neg[:, 1].copy()
                AV_pos[:, 3*96] = -VR_abc_neg[:, 2].copy()
                bV_pos = VR_abc_pos[:, 0]

                A = np.vstack([A1, A2, A3, A4, AV_neg, AV_pos])
                b = np.concatenate([b1, b2, b3, b4, bV_neg, bV_pos])

                lb = np.concatenate([np.zeros(96), -np.inf*np.ones(96), np.zeros(96*10), -x_max_turbine*np.ones(24), np.full(1, -np.inf)])
                ub = np.concatenate([Rmax*np.ones(96), np.inf*np.ones(96*7), np.ones(96*4), x_max_pump*np.ones(24), np.full(1, np.inf)])

                intcon = np.arange(8*96, 96*10)

                # Specify bounds (lower and upper bounds)
                bounds = Bounds(lb, ub)
                # bounds = [(lb[i], ub[i]) for i in range(len(lb))]
                # Specify inequality constraints (Ax <= b)
                ineq_constraints = LinearConstraint(A, -float('inf'), b)
                # Specify equality constraints (Aeq x = beq)
                eq_constraints = LinearConstraint(Aeq, beq, beq)
                # Define the integer variable indices
                integrality = [1 if i in intcon else 0 for i in range(len(f))]

                # Solve the MILP problem
                result = milp(c=-f, constraints=[ineq_constraints, eq_constraints], bounds=bounds,
                              integrality=integrality
                              )
                # result = linprog(
                #     c=-f,            # Coefficients for the objective function (minimize f^T * x)
                #     A_ub=A,         # Inequality constraint matrix
                #     b_ub=b,         # Inequality constraint bounds
                #     A_eq=Aeq,       # Equality constraint matrix
                #     b_eq=beq,       # Equality constraint bounds
                #     bounds=bounds,  # Bounds on the decision variables
                #     method='highs' # Use the 'highs' method (default for modern solvers in scipy)
                    # integrality=integrality
                # )

                # Extract the objective function value (fval) from the result
                fval = result.fun

                # Update the Vt matrix with the result (assuming `iR, ix, n, t_i` are indices)
                Vt[iR, ix, n, t_i] = -fval
print(Vt)

# Forward simulation (similar complexity to backward step)
np.random.seed(seed + 1)

sample_P_day_all = np.zeros((N, T, 24*D))
sample_P_intraday_all = np.zeros((N, T, 96*D))

Wt_day_mat = np.zeros((M, T*24))
Wt_intra_mat = np.zeros((M, T*96))

for n in range(M):
    P_day = P_day_0[:24*D].copy()
    P_intraday = P_intraday_0[:96*D].copy()
    for t_i in range(T):
        sample_P_day_all[n, t_i, :] = P_day
        sample_P_intraday_all[n, t_i, :] = P_intraday

        mu_day, cor_day = sample_price_day(P_day, t_i, Season)
        Wt_day = multivariate_normal.rvs(mean=mu_day, cov=cor_day)

        mu_intraday, cor_intraday = sample_price_intraday(np.concatenate([Wt_day, P_day]), P_intraday, t_i, Season)
        Wt_intraday = multivariate_normal.rvs(mean=mu_intraday, cov=cor_intraday)

        P_day = np.concatenate([Wt_day, P_day[:-24].copy()])
        P_intraday = np.concatenate([Wt_intraday, P_intraday[:-96].copy()])

        Wt_day_mat[n, t_i*24:(t_i+1)*24] = Wt_day
        Wt_intra_mat[n, t_i*96:(t_i+1)*96] = Wt_intraday

R_0=0
x0_0=0

V=np.zeros((M,1))

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
    P_day = P_day_0[:24 * D].copy()
    P_intraday = P_intraday_0[:96 * D].copy()

    P_day_sim = P_day_0[:24 * D].copy()
    P_intraday_sim = P_intraday_0[:96 * D].copy()

    C = 0

    for t_i in range(T):
        mu_day, _ = sample_price_day(P_day_sim, t_i, Season)
        mu_intraday, _ = sample_price_intraday(np.concatenate([mu_day, P_day_sim]), P_intraday_sim, t_i, Season)


        P_day_next = np.concatenate([mu_day, P_day[:-24].copy()])
        P_intraday_next = np.concatenate([mu_intraday, P_intraday[:-96].copy()])

        lk = 2
        VR_abc_neg = np.zeros((lk - 1, 3))
        VR_abc_pos = np.zeros((lk - 1, 3))

        if t_i < T-1 and np.any(Vt != 0):
            P_day_sample_next = P_day_state[:, t_i + 1, :].copy().reshape(N, D * 24)
            P_intraday_sample_next = P_intra_state[:, t_i + 1, :].copy().reshape(N, D * 24 * 4)

            phi = np.hstack((P_day_sample_next, P_intraday_sample_next))
            Y = np.hstack((P_day_next, P_intraday_next))

            # Update weights using the VRx_weights function
            weights = VRx_weights(phi, Y, weights_D_value[t_i + 1, :].copy())

            VRx = np.zeros((length_R, 3))
            for i in range(length_R):
                for j in range(3):
                    VRx[i, j] = Vt[i, j, :, t_i+1].copy().reshape(1, N) @ weights

            # Create input for convex hull computation
            hull_input = np.column_stack([R_vec.T, VRx[:, 1]])

            # Compute the convex hull
            hull = ConvexHull(hull_input)
            k = hull.vertices

            # Sort k by R_vec values to ensure ordering consistency
            # k = k[np.argsort(R_vec[k])]

            # Remove the first element (equivalent to `k(1)=[]` in MATLAB)
            # k = k[1:]
            k = np.sort(k)[::-1]

            # Length of k
            lk = len(k)
            VR_abc_neg = np.zeros((lk - 1, 3))
            VR_abc_pos = np.zeros((lk - 1, 3))

            for i in range(1, lk):  # MATLAB indexing starts from 2; Python indexing starts from 1
                VR_abc_neg[i - 1, 1] = (VR[i, 1] - VR[i - 1, 1]) / (R_k[i] - R_k[i - 1])  # Steigung R
                VR_abc_neg[i - 1, 0] = VR[i, 1] - VR_abc_neg[i - 1, 1] * R_k[i]  # Achsenabschnitt
                VR_abc_neg[i - 1, 2] = -(VR[i - 1, 1] - VR[i - 1, 0]) / (x_vec[1] - x_vec[0])  # Steigung x

            # Loop for VR_abc_pos
            for i in range(1, lk):  # MATLAB indexing starts from 2; Python indexing starts from 1
                VR_abc_pos[i - 1, 1] = (VR[i, 1] - VR[i - 1, 1]) / (R_k[i] - R_k[i - 1])  # Steigung R
                VR_abc_pos[i - 1, 0] = VR[i, 1] - VR_abc_pos[i - 1, 1] * R_k[i]  # Achsenabschnitt
                VR_abc_pos[i - 1, 2] = (VR[i - 1, 1] - VR[i - 1, 2]) / (x_vec[1] - x_vec[2])  # Steigung x

        f = np.zeros(96 * 12 + 24 + 1)
        f[-1] = 1

        f[96:192] = f[96:192] - Delta_ti * mu_intraday
        f[-25:-1] = -Delta_td * mu_day

        q_pump_up = (np.abs(mu_intraday) / Q_mult - Q_fix) * t_ramp_pump_up / 2
        q_pump_down = (np.abs(mu_intraday) * Q_mult + Q_fix) * t_ramp_pump_down / 2
        q_turbine_up = (np.abs(mu_intraday) * Q_mult + Q_fix) * t_ramp_turbine_up / 2
        q_turbine_down = (np.abs(mu_intraday) / Q_mult - Q_fix) * t_ramp_turbine_down / 2

        # Grid fee
        f[96 * 2:96 * 3] = f[96 * 2:96 * 3] - c_grid_fee
        # Delta pump up
        f[96 * 4:96 * 5] = f[96 * 4:96 * 5] + q_pump_up
        # Delta pump down
        f[96 * 5:96 * 6] = f[96 * 5:96 * 6] - q_pump_down
        # Delta turbine up
        f[96 * 6:96 * 7] = f[96 * 6:96 * 7] - q_turbine_up
        # Delta turbine down
        f[96 * 7:96 * 8] = f[96 * 7:96 * 8] + q_turbine_down

        # z^pump
        f[96 * 10:96 * 11] = f[96 * 10:96 * 11] - Q_start_pump
        # z^turbine
        f[96 * 11:96 * 12] = f[96 * 11:96 * 12] - Q_start_turbine

        A1 = np.hstack([
            -np.eye(96) + np.diag(np.ones(95), -1),
            np.zeros((96, 96)),
            Delta_ti * beta_pump * np.eye(96),
            -Delta_ti / beta_turbine * np.eye(96),
            -beta_pump * c_pump_up * np.eye(96),
            beta_pump * c_pump_down * np.eye(96),
            c_turbine_up / beta_turbine * np.eye(96),
            -c_turbine_down / beta_turbine * np.eye(96),
            np.zeros((96, 96 * 4 + 24)),
            np.zeros((96, 1))
        ])
        b1 = np.zeros(96)
        b1[0] = -R

        # A2
        Axh = np.zeros((96, 24))
        for h in range(24):
            Axh[h * 4:(h + 1) * 4, h] = -1

        A2 = np.hstack([
            np.zeros((96, 96)),
            -np.eye(96),
            np.eye(96),
            -np.eye(96),
            np.zeros((96, 96 * 8)),
            Axh,
            np.zeros((96, 1))
        ])
        b2 = np.zeros(96)

        A3 = np.hstack([
            np.zeros((96, 96*2)),
            np.eye(96) - np.diag(np.ones(95), -1),
            np.zeros((96, 96)),
            -np.eye(96),
            np.eye(96),
            np.zeros((96, 96*6 + 24)),
            np.zeros((96, 1))
        ])
        b3 = np.zeros(96)
        b3[0] = max(x0, 0)

        A4 = np.hstack([
            np.zeros((96, 96*3)),
            np.eye(96) - np.diag(np.ones(95), -1),
            np.zeros((96, 96*2)),
            -np.eye(96),
            np.eye(96),
            np.zeros((96, 96*4 + 24)),
            np.zeros((96, 1))
        ])
        b4=np.zeros(96)
        b4[0]=max(-x0,0)

        Aeq = np.vstack([A1, A2, A3, A4])
        beq = np.hstack([b1, b2, b3, b4])

        A1 = np.vstack([
            np.hstack([np.zeros((96, 96*2)), -np.eye(96), np.zeros((96, 96*5)), x_min_pump * np.eye(96), np.zeros((96, 96*3+24)), np.zeros((96, 1))]),
            np.hstack([np.zeros((96, 96*2)), np.eye(96), np.zeros((96, 96*5)), -x_max_pump * np.eye(96), np.zeros((96, 96*3+24)), np.zeros((96, 1))]),
            np.hstack([np.zeros((96, 96*3)), -np.eye(96), np.zeros((96, 96*5)), x_min_turbine * np.eye(96), np.zeros((96, 96*2+24)), np.zeros((96, 1))]),
            np.hstack([np.zeros((96, 96*3)), np.eye(96), np.zeros((96, 96*5)), -x_max_turbine * np.eye(96), np.zeros((96, 96*2+24)), np.zeros((96, 1))])
        ])

        b1=np.zeros(96*4)

        A2 = np.hstack([
            np.zeros((96, 96*8)),
            np.eye(96) - np.diag(np.ones(95), -1),
            np.zeros((96, 96)),
            -np.eye(96),
            np.zeros((96, 96 + 24)),
            np.zeros((96, 1))
        ])

        # Construct b2
        b2 = np.zeros(96)
        b2[0] = x0 > 0  # equivalent to b2(1) = x0 > 0 in MATLAB

        # Construct A3
        A3 = np.hstack([
            np.zeros((96, 96*9)),
            np.eye(96) - np.diag(np.ones(95), -1),
            np.zeros((96, 96)),
            -np.eye(96),
            np.zeros((96, 24)),
            np.zeros((96, 1))
        ])

        # Construct b3
        b3 = np.zeros(96)
        b3[0] = x0 < 0  # equivalent to b3(1) = x0 < 0 in MATLAB

        A4 = np.hstack([
            np.zeros((96, 96*8)),
            np.eye(96),
            np.eye(96),
            np.zeros((96, 2*96+24)),
            np.zeros((96, 1))
        ])

        # Construct b4
        b4 = np.ones(96)

        AV_neg = np.zeros((lk-1, 12*96 + 24 + 1))
        AV_neg[:, -1] = 1
        AV_neg[:, 96] = -VR_abc_neg[:, 1].copy()
        AV_neg[:, 4*96] = -VR_abc_neg[:, 2].copy()
        bV_neg = VR_abc_neg[:, 0].copy()

        AV_pos = np.zeros((lk-1, 12*96 + 24 + 1))
        AV_pos[:, -1] = 1
        AV_pos[:, 96] = -VR_abc_neg[:, 1].copy()
        AV_pos[:, 3*96] = -VR_abc_neg[:, 2].copy()
        bV_pos = VR_abc_pos[:, 0].copy()

        A = np.vstack([A1, A2, A3, A4, AV_neg, AV_pos])
        b = np.concatenate([b1, b2, b3, b4, bV_neg, bV_pos])

        lb = np.concatenate([np.zeros(96), -np.inf*np.ones(96), np.zeros(96*10), -x_max_turbine*np.ones(24), np.full(1, -np.inf)])
        ub = np.concatenate([Rmax*np.ones(96), np.inf*np.ones(96*7), np.ones(96*4), x_max_pump*np.ones(24), np.full(1, np.inf)])

        intcon = np.arange(8*96, 96*10)

        # Specify bounds (lower and upper bounds)
        bounds = Bounds(lb, ub)
        # bounds = [(lb[i], ub[i]) for i in range(len(lb))]
        # Specify inequality constraints (Ax <= b)
        ineq_constraints = LinearConstraint(A, -float('inf'), b)
        # Specify equality constraints (Aeq x = beq)
        eq_constraints = LinearConstraint(Aeq, beq, beq)
        # Define the integer variable indices
        integrality = [1 if i in intcon else 0 for i in range(len(f))]

        # Solve the MILP problem
        result = milp(c=-f, constraints=[ineq_constraints, eq_constraints], bounds=bounds,
                      integrality=integrality
                      )
        # result = linprog(
        #     c=-f,            # Coefficients for the objective function (minimize f^T * x)
        #     A_ub=A,         # Inequality constraint matrix
        #     b_ub=b,         # Inequality constraint bounds
        #     A_eq=Aeq,       # Equality constraint matrix
        #     b_eq=beq,       # Equality constraint bounds
        #     bounds=bounds,  # Bounds on the decision variables
        #     method='highs' # Use the 'highs' method (default for modern solvers in scipy)
            # integrality=integrality
        # )
        # Extract the optimized values
        x_opt = result.x
        # Extract the last 24 values (equivalent to xday_opt)
        xday_opt = x_opt[-24:].copy()

        # Extract Wt_day from Wt_day_mat
        Wt_day = Wt_day_mat[m, t_i*24:(t_i+1)*24].copy()
        day_path = np.tile(Wt_day, (4, 1))
        P_day_path[m, t_i*96:(t_i+1)*96] = day_path.flatten()

        mu_intraday, _ = sample_price_intraday(np.concatenate([Wt_day, P_day_sim]), P_intraday_sim, t_i, Season)

        P_day_next = np.concatenate([Wt_day, P_day[:-24].copy()])
        P_intraday_next = np.concatenate([mu_intraday, P_intraday[:-96].copy()])

        lk = 2
        VR_abc_neg = np.zeros((lk - 1, 3))
        VR_abc_pos = np.zeros((lk - 1, 3))

        if t_i < T-1 and np.any(Vt != 0):
            phi = np.hstack((P_day_sample_next, P_intraday_sample_next))
            Y = np.hstack((P_day_next, P_intraday_next))

            weights = VRx_weights(phi, Y, weights_D_value[t_i])

            # Initialize VRx as a NumPy array of zeros with shape (length_R, 3)
            VRx = np.zeros((length_R, 3))

            # Loop over length_R and 3 to compute the weighted sum for VRx
            for i in range(length_R):
                for j in range(3):
                    # Reshape Vt[i, j, :, t+1] to a 1D array and compute the dot product with weights
                    VRx[i, j] = Vt[i, j, :, t_i+1].copy().reshape(1, N) @ weights

            lk = len(k)
            VR_abc_neg = np.zeros((lk - 1, 3))
            VR_abc_pos = np.zeros((lk - 1, 3))
            VR = VRx[k, :]
            R_k = R_vec[k]

            for i in range(1, lk):  # MATLAB indexing starts from 2; Python indexing starts from 1
                VR_abc_neg[i - 1, 1] = (VR[i, 1] - VR[i - 1, 1]) / (R_k[i] - R_k[i - 1])  # Steigung R
                VR_abc_neg[i - 1, 0] = VR[i, 1] - VR_abc_neg[i - 1, 1] * R_k[i]  # Achsenabschnitt
                VR_abc_neg[i - 1, 2] = -(VR[i - 1, 1] - VR[i - 1, 0]) / (x_vec[1] - x_vec[0])  # Steigung x

            # Loop for VR_abc_pos
            for i in range(1, lk):  # MATLAB indexing starts from 2; Python indexing starts from 1
                VR_abc_pos[i - 1, 1] = (VR[i, 1] - VR[i - 1, 1]) / (R_k[i] - R_k[i - 1])  # Steigung R
                VR_abc_pos[i - 1, 0] = VR[i, 1] - VR_abc_pos[i - 1, 1] * R_k[i]  # Achsenabschnitt
                VR_abc_pos[i - 1, 2] = (VR[i - 1, 1] - VR[i - 1, 2]) / (x_vec[1] - x_vec[2])  # Steigung x


        f = np.zeros(96 * 12 + 24 + 1)
        f[-1] = 1

        f[96:192] = f[96:192] - Delta_ti * mu_intraday
        f[-25:-1] = -Delta_td * mu_day

        q_pump_up = (np.abs(mu_intraday) / Q_mult - Q_fix) * t_ramp_pump_up / 2
        q_pump_down = (np.abs(mu_intraday) * Q_mult + Q_fix) * t_ramp_pump_down / 2
        q_turbine_up = (np.abs(mu_intraday) * Q_mult + Q_fix) * t_ramp_turbine_up / 2
        q_turbine_down = (np.abs(mu_intraday) / Q_mult - Q_fix) * t_ramp_turbine_down / 2

        # Grid fee
        f[96 * 2:96 * 3] = f[96 * 2:96 * 3] - c_grid_fee
        # Delta pump up
        f[96 * 4:96 * 5] = f[96 * 4:96 * 5] + q_pump_up
        # Delta pump down
        f[96 * 5:96 * 6] = f[96 * 5:96 * 6] - q_pump_down
        # Delta turbine up
        f[96 * 6:96 * 7] = f[96 * 6:96 * 7] - q_turbine_up
        # Delta turbine down
        f[96 * 7:96 * 8] = f[96 * 7:96 * 8] + q_turbine_down

        # z^pump
        f[96 * 10:96 * 11] = f[96 * 10:96 * 11] - Q_start_pump
        # z^turbine
        f[96 * 11:96 * 12] = f[96 * 11:96 * 12] - Q_start_turbine

        A1 = np.hstack([
            -np.eye(96) + np.diag(np.ones(95), -1),
            np.zeros((96, 96)),
            Delta_ti * beta_pump * np.eye(96),
            -Delta_ti / beta_turbine * np.eye(96),
            -beta_pump * c_pump_up * np.eye(96),
            beta_pump * c_pump_down * np.eye(96),
            c_turbine_up / beta_turbine * np.eye(96),
            -c_turbine_down / beta_turbine * np.eye(96),
            np.zeros((96, 96 * 4 + 24)),
            np.zeros((96, 1))
        ])
        b1 = np.zeros(96)
        b1[0] = -R

        # A2
        Axh = np.zeros((96, 24))
        for h in range(24):
            Axh[h * 4:(h + 1) * 4, h] = -1

        A2 = np.hstack([
            np.zeros((96, 96)),
            -np.eye(96),
            np.eye(96),
            -np.eye(96),
            np.zeros((96, 96 * 8)),
            Axh,
            np.zeros((96, 1))
        ])
        b2 = np.zeros(96)

        A3 = np.hstack([
            np.zeros((96, 96*2)),
            np.eye(96) - np.diag(np.ones(95), -1),
            np.zeros((96, 96)),
            -np.eye(96),
            np.eye(96),
            np.zeros((96, 96*6 + 24)),
            np.zeros((96, 1))
        ])
        b3 = np.zeros(96)
        b3[0] = max(x0, 0)

        A4 = np.hstack([
            np.zeros((96, 96*3)),
            np.eye(96) - np.diag(np.ones(95), -1),
            np.zeros((96, 96*2)),
            -np.eye(96),
            np.eye(96),
            np.zeros((96, 96*4 + 24)),
            np.zeros((96, 1))
        ])
        b4=np.zeros(96)
        b4[0]=max(-x0,0)

        Aeq = np.vstack([A1, A2, A3, A4])
        beq = np.hstack([b1, b2, b3, b4])

        A1 = np.vstack([
            np.hstack([np.zeros((96, 96*2)), -np.eye(96), np.zeros((96, 96*5)), x_min_pump * np.eye(96), np.zeros((96, 96*3+24)), np.zeros((96, 1))]),
            np.hstack([np.zeros((96, 96*2)), np.eye(96), np.zeros((96, 96*5)), -x_max_pump * np.eye(96), np.zeros((96, 96*3+24)), np.zeros((96, 1))]),
            np.hstack([np.zeros((96, 96*3)), -np.eye(96), np.zeros((96, 96*5)), x_min_turbine * np.eye(96), np.zeros((96, 96*2+24)), np.zeros((96, 1))]),
            np.hstack([np.zeros((96, 96*3)), np.eye(96), np.zeros((96, 96*5)), -x_max_turbine * np.eye(96), np.zeros((96, 96*2+24)), np.zeros((96, 1))])
        ])

        b1=np.zeros(96*4)

        A2 = np.hstack([
            np.zeros((96, 96*8)),
            np.eye(96) - np.diag(np.ones(95), -1),
            np.zeros((96, 96)),
            -np.eye(96),
            np.zeros((96, 96 + 24)),
            np.zeros((96, 1))
        ])

        # Construct b2
        b2 = np.zeros(96)
        b2[0] = x0 > 0

        # Construct A3
        A3 = np.hstack([
            np.zeros((96, 96*9)),
            np.eye(96) - np.diag(np.ones(95), -1),
            np.zeros((96, 96)),
            -np.eye(96),
            np.zeros((96, 24)),
            np.zeros((96, 1))
        ])

        # Construct b3
        b3 = np.zeros(96)
        b3[0] = x0 < 0

        A4 = np.hstack([
            np.zeros((96, 96*8)),
            np.eye(96),
            np.eye(96),
            np.zeros((96, 2*96+24)),
            np.zeros((96, 1))
        ])

        # Construct b4
        b4 = np.ones(96)

        AV_neg = np.zeros((lk-1, 12*96 + 24 + 1))
        AV_neg[:, -1] = 1
        AV_neg[:, 96] = -VR_abc_neg[:, 1].copy()
        AV_neg[:, 4*96] = -VR_abc_neg[:, 2].copy()
        bV_neg = VR_abc_neg[:, 0].copy()

        AV_pos = np.zeros((lk-1, 12*96 + 24 + 1))
        AV_pos[:, -1] = 1
        AV_pos[:, 96] = -VR_abc_neg[:, 1].copy()
        AV_pos[:, 3*96] = -VR_abc_neg[:, 2].copy()
        bV_pos = VR_abc_pos[:, 0].copy()

        A = np.vstack([A1, A2, A3, A4, AV_neg, AV_pos])
        b = np.concatenate([b1, b2, b3, b4, bV_neg, bV_pos])

        lb = np.concatenate([np.zeros(96), -np.inf*np.ones(96), np.zeros(96*10), -x_max_turbine*np.ones(24), np.full(1, -np.inf)])
        ub = np.concatenate([Rmax*np.ones(96), np.inf*np.ones(96*7), np.ones(96*4), x_max_pump*np.ones(24), np.full(1, np.inf)])

        intcon = np.arange(8*96, 96*10)

        # Specify bounds (lower and upper bounds)
        bounds = Bounds(lb, ub)
        # bounds = [(lb[i], ub[i]) for i in range(len(lb))]
        # Specify inequality constraints (Ax <= b)
        ineq_constraints = LinearConstraint(A, -float('inf'), b)
        # Specify equality constraints (Aeq x = beq)
        eq_constraints = LinearConstraint(Aeq, beq, beq)
        # Define the integer variable indices
        integrality = [1 if i in intcon else 0 for i in range(len(f))]

        # Solve the MILP problem
        result = milp(c=-f, constraints=[ineq_constraints, eq_constraints], bounds=bounds
                    #   integrality=integrality
                      )
        # result = linprog(
        #     c=-f,            # Coefficients for the objective function (minimize f^T * x)
        #     A_ub=A,         # Inequality constraint matrix
        #     b_ub=b,         # Inequality constraint bounds
        #     A_eq=Aeq,       # Equality constraint matrix
        #     b_eq=beq,       # Equality constraint bounds
        #     bounds=bounds,  # Bounds on the decision variables
        #     method='highs' # Use the 'highs' method (default for modern solvers in scipy)
            # integrality=integrality
        # )

        x_opt = result.x

        # Extract slices from x_opt
        R_opt = x_opt[:96].copy()  # Equivalent to x_opt(1:96)
        xhq_opt = x_opt[96:2*96].copy()  # Equivalent to x_opt(1+96:2*96)

        Delta_pump_up = x_opt[4*96:5*96].copy()  # Equivalent to x_opt(4*96+1:5*96)
        Delta_pump_down = x_opt[5*96:6*96].copy()  # Equivalent to x_opt(5*96+1:6*96)
        Delta_turbine_up = x_opt[6*96:7*96].copy() # Equivalent to x_opt(6*96+1:7*96)
        Delta_turbine_down = x_opt[7*96:8*96].copy()  # Equivalent to x_opt(7*96+1:8*96)

        # Extract slices from x_opt
        x_pump = x_opt[2*96:3*96].copy()  # Equivalent to x_opt(2*96+1:3*96)
        x_turbine = x_opt[3*96:4*96].copy()  # Equivalent to x_opt(3*96+1:4*96)
        y_pump = x_opt[8*96:9*96].copy()  # Equivalent to x_opt(8*96+1:9*96)
        y_turbine = x_opt[9*96:10*96].copy()  # Equivalent to x_opt(9*96+1:10*96)
        z_pump = x_opt[10*96:11*96].copy()  # Equivalent to x_opt(10*96+1:11*96)
        z_turbine = x_opt[11*96:12*96].copy()  # Equivalent to x_opt(11*96+1:12*96)

        # Assuming `R_path`, `x_intraday_path`, `x_pump_path`, etc. are initialized as numpy arrays with the correct shape
        # Assign values to the paths
        R_path[m, t_i*96:96*(t_i+1)] = R_opt.T
        x_intraday_path[m, t_i*96:96*(t_i+1)] = xhq_opt.T

        x_pump_path[m, t_i*96:96*(t_i+1)] = x_pump.T
        x_turbine_path[m, t_i*96:96*(t_i+1)] = x_turbine.T
        y_pump_path[m, t_i*96:96*(t_i+1)] = y_pump.T
        y_turbine_path[m, t_i*96:96*(t_i+1)] = y_turbine.T
        z_pump_path[m, t_i*96:96*(t_i+1)] = z_pump
        z_turbine_path[m, t_i*96:96*(t_i+1)] = z_turbine.T

        # Extract Wt_intraday from Wt_intra_mat
        Wt_intraday = Wt_intra_mat[m, t_i*96:96*(t_i+1)].copy()
        P_intraday_path[m, t_i*96:96*(t_i+1)] = Wt_intraday

        # Calculate the values for q_pump_up, q_pump_down, q_turbine_up, q_turbine_down
        q_pump_up = (np.abs(Wt_intraday) / Q_mult - Q_fix) * t_ramp_pump_up / 2
        q_pump_down = (np.abs(Wt_intraday) * Q_mult + Q_fix) * t_ramp_pump_down / 2
        q_turbine_up = (np.abs(Wt_intraday) * Q_mult + Q_fix) * t_ramp_turbine_up / 2
        q_turbine_down = (np.abs(Wt_intraday) / Q_mult - Q_fix) * t_ramp_turbine_down / 2

        # Extract values from R_opt and x_pump, x_turbine
        R = R_opt[-1].copy()  # Last element in R_opt
        x0 = x_pump[-1].copy() - x_turbine[-1].copy()  # Last element of x_pump minus last element of x_turbine

        # Update P_day, P_intraday, P_day_sim, P_intraday_sim
        P_day = np.concatenate((Wt_day, P_day[:-24].copy()))  # Concatenate Wt_day with the first part of P_day excluding last 24
        P_intraday = np.concatenate((Wt_intraday, P_intraday[:-96].copy()))  # Concatenate Wt_intraday with the first part of P_intraday excluding last 96
        P_day_sim = np.concatenate((Wt_day, P_day_sim[:-24].copy()))  # Same for P_day_sim
        P_intraday_sim = np.concatenate((Wt_intraday, P_intraday_sim[:-96].copy()))  # Same for P_intraday_sim

        # Update C with the given formula
        C = C - Delta_td * np.dot(Wt_day, xday_opt) - np.sum(x_pump) * c_grid_fee \
            - Delta_ti * np.dot(Wt_intraday, xhq_opt) + \
            np.dot(q_pump_up, Delta_pump_up) - np.dot(q_pump_down, Delta_pump_down) - \
            np.dot(q_turbine_up, Delta_turbine_up) + np.dot(q_turbine_down, Delta_turbine_down) - \
            np.sum(z_pump) * Q_start_pump - np.sum(z_turbine) * Q_start_turbine

    V[m] = C

EV = np.mean(V)
print(EV)
