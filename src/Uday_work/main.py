import numpy as np
from scipy.io import loadmat
from scipy.stats import multivariate_normal
from scipy.spatial import ConvexHull
import os
import warnings
import matlab.engine

# If these imports are local, ensure they are available:
from sample_price_day import sample_price_day
from sample_price_intraday import sample_price_intraday

warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

N = 50
M = 10
T = 3
Season = 'Summer'
length_R = 5
seed = 2

D = 7  # days in forecast
np.random.seed(seed)
Rmax = 100

# Start MATLAB engine
eng = matlab.engine.start_matlab()

weights_D_value = eng.badp_weights(T)
weights_D_value = np.array(weights_D_value)

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

Vt = np.zeros((length_R, 3, N, T+1))

# Create intlinprog options in MATLAB
options = eng.optimoptions('intlinprog', 'Display', 'off')

# Backward step
for t_i in range(T-1, -1, -1):
    P_day_sample = P_day_state[:, t_i, :].copy()
    P_intraday_sample = P_intra_state[:, t_i, :].copy()

    if t_i < T-1:
        P_day_sample_next = P_day_state[:, t_i + 1, :].reshape(N, D * 24)
        P_intraday_sample_next = P_intra_state[:, t_i + 1, :].reshape(N, D * 24 * 4)

    for n in range(N):
        P_day = P_day_sample[n, :].copy()
        P_intraday = P_intraday_sample[n, :].copy()

        mu_day, cor_day = sample_price_day(P_day, t_i, Season)
        mu_intraday, cor_intraday = sample_price_intraday(np.concatenate([mu_day, P_day]), P_intraday, t_i, Season)

        P_day_next = np.concatenate([mu_day, P_day[:-24]])
        P_intraday_next = np.concatenate([mu_intraday, P_intraday[:-96]])

        lk = 2
        VR_abc_neg = np.zeros((lk-1, 3))
        VR_abc_pos = np.zeros((lk-1, 3))

        if t_i < T-1:
            phi = np.concatenate([P_day_sample_next, P_intraday_sample_next], axis=1)
            Y = np.concatenate([P_day_next, P_intraday_next])

            # Convert phi, Y to MATLAB arrays
            phi_mat = matlab.double(phi.tolist())
            Y_mat = matlab.double(Y.tolist())
            weights_row = matlab.double(weights_D_value[int(t_i+1), :].tolist())
            weights_mat = eng.VRx_weights(phi_mat, Y_mat, weights_row)
            weights = np.array(weights_mat).flatten()

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
            # Compute VR_abc_neg and VR_abc_pos (same logic as in the original code)
            for i in range(1, lk):
                VR_abc_neg[i - 1, 1] = (VR[i, 1] - VR[i - 1, 1]) / (R_k[i] - R_k[i - 1])
                VR_abc_neg[i - 1, 0] = VR[i, 1] - VR_abc_neg[i - 1, 1]*R_k[i]
                VR_abc_neg[i - 1, 2] = -(VR[i - 1, 1] - VR[i - 1, 0])/(x_vec[1]-x_vec[0])

            for i in range(1, lk):
                VR_abc_pos[i - 1, 1] = (VR[i, 1] - VR[i - 1, 1])/(R_k[i]-R_k[i-1])
                VR_abc_pos[i - 1, 0] = VR[i, 1] - VR_abc_pos[i - 1, 1]*R_k[i]
                VR_abc_pos[i - 1, 2] = (VR[i-1, 1] - VR[i-1, 2])/(x_vec[1]-x_vec[2])

        # Solve MILP for each state (R, x0)
        for iR in range(length_R):
            R_val = R_vec[iR]
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
                b1[0] = -R_val

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

                # Convert all arrays to MATLAB doubles
                matlab_f = matlab.double((-f).tolist())  # since we minimize -f, we pass f as negative
                matlab_A = matlab.double(A.tolist())
                matlab_b = matlab.double(b.tolist())
                matlab_Aeq = matlab.double(Aeq.tolist())
                matlab_beq = matlab.double(beq.tolist())
                matlab_lb = matlab.double(lb.tolist())
                matlab_ub = matlab.double(ub.tolist())

                # Convert intcon to 1-based indexing
                matlab_intcon = matlab.double((intcon+1).tolist())

                # Call intlinprog
                xres, fvalres = eng.intlinprog(matlab_f, matlab_intcon,
                                            matlab_A, matlab_b,
                                            matlab_Aeq, matlab_beq,
                                            matlab_lb, matlab_ub,
                                            options, nargout=2)


                x_opt = np.array(xres).flatten()  # Convert solution vector to a NumPy array
                fval = float(fvalres)             # Convert fvalres to a float


                Vt[iR, ix, n, t_i] = -fval


Vt
