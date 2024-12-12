# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import matlab.engine
from scipy.spatial import ConvexHull

# Import local modules
import config  # Import the configuration module
from badp_weights import badp_weights
from sample_price_day import sample_price_day
from sample_price_intraday import sample_price_intraday
from VRx_weights import VRx_weights

# Start MATLAB engine
eng = matlab.engine.start_matlab()
# %%


def compute_EV(N, M, T, Season, length_R, seed, config_data):
    # %%
    """
    Compute expected value (EV) using parameters from config_data.
    """
    # for debugging
    config_data = config.load_config(season="Summer")
    length_R = config_data["length_R"]
    N = config_data["N"]
    T = config_data["T"]
    M = config_data["M"]
    seed = config_data["seed"]
    Season = "Summer"

    np.random.seed(seed)

    # Extract parameters from config_data
    D = config_data["D"]
    Rmax = config_data["Rmax"]
    P_day_0 = config_data["P_day_0"]
    P_intraday_0 = config_data["P_intraday_0"]

    t_ramp_pump_up = config_data["t_ramp_pump_up"]
    t_ramp_pump_down = config_data["t_ramp_pump_down"]
    t_ramp_turbine_up = config_data["t_ramp_turbine_up"]
    t_ramp_turbine_down = config_data["t_ramp_turbine_down"]
    c_grid_fee = config_data["c_grid_fee"]
    Delta_ti = config_data["Delta_ti"]
    Delta_td = config_data["Delta_td"]
    Q_mult = config_data["Q_mult"]
    Q_fix = config_data["Q_fix"]
    Q_start_pump = config_data["Q_start_pump"]
    Q_start_turbine = config_data["Q_start_turbine"]
    beta_pump = config_data["beta_pump"]
    beta_turbine = config_data["beta_turbine"]
    x_max_pump = config_data["x_max_pump"]
    x_min_pump = config_data["x_min_pump"]
    x_max_turbine = config_data["x_max_turbine"]
    x_min_turbine = config_data["x_min_turbine"]
    R_vec = config_data["R_vec"]
    x_vec = config_data["x_vec"]
    c_pump_up = config_data["c_pump_up"]
    c_pump_down = config_data["c_pump_down"]
    c_turbine_up = config_data["c_turbine_up"]
    c_turbine_down = config_data["c_turbine_down"]

    weights_D_value = config_data["weights_D_value"]

    # %%

    # Set MILP options
    # Set RNG in both Python and MATLAB
    eng.rng(seed, nargout=0)

    intlinprog_options = eng.optimoptions("intlinprog", "Display", "off")

    # %%

    # Dimensions
    # Day-ahead: 24 hours * D days
    # Intraday: 96 quarter-hours * D days (4 quarter-hours per hour * 24 hours * D days)

    # Sample path
    sample_P_day_all = np.zeros((N, T, 24 * D))
    sample_P_intraday_all = np.zeros((N, T, 4 * 24 * D))

    Wt_day_mat = np.zeros((N, T * 24))
    Wt_intra_mat = np.zeros((N, T * 96))

    # Generate sample paths
    for n in range(N):
        # Initialize P_day and P_intraday
        P_day = P_day_0[: 24 * D].copy()
        P_intraday = P_intraday_0[: 96 * D].copy()

        for t in range(T):
            # Store current paths
            sample_P_day_all[n, t, :] = P_day
            sample_P_intraday_all[n, t, :] = P_intraday

            # Compute mu_day, cor_day
            mu_day, cor_day = sample_price_day(P_day, t + 1, Season)
            Wt_day = np.random.multivariate_normal(mu_day, cor_day)

            # Compute mu_intraday, cor_intraday
            mu_intraday, cor_intraday = sample_price_intraday(
                np.concatenate([Wt_day, P_day]), P_intraday, t + 1, Season
            )
            Wt_intraday = np.random.multivariate_normal(mu_intraday, cor_intraday)

            # Update P_day and P_intraday
            P_day = np.concatenate([Wt_day, P_day[:-24]])
            P_intraday = np.concatenate([Wt_intraday, P_intraday[:-96]])

            # Store Wt_day and Wt_intraday
            Wt_day_mat[n, t * 24 : (t + 1) * 24] = Wt_day
            Wt_intra_mat[n, t * 96 : (t + 1) * 96] = Wt_intraday

    # In MATLAB:
    # P_day_state=sample_P_day_all;
    # P_intra_state=sample_P_intraday_all;
    # Here we just return them
    P_day_state = sample_P_day_all
    P_intra_state = sample_P_intraday_all

    # Lookup Table Vt
    Vt = np.zeros((length_R, 3, N, T + 1))

    for t_idx in range(T - 1, -1, -1):
        P_day_sample = P_day_state[:, t_idx, :].reshape(N, D * 24)
        P_intraday_sample = P_intra_state[:, t_idx, :].reshape(N, D * 24 * 4)

        if t_idx < T - 1:
            P_day_sample_next = P_day_state[:, t_idx + 1, :].reshape(N, D * 24)
            P_intraday_sample_next = P_intra_state[:, t_idx + 1, :].reshape(
                N, D * 24 * 4
            )

        for n_i in range(N):
            P_day = P_day_sample[n_i, :]
            P_intraday = P_intraday_sample[n_i, :]

            mu_day_res = sample_price_day(P_day, t_idx + 1, Season)
            if isinstance(mu_day_res, tuple):
                mu_day, _ = mu_day_res
            else:
                mu_day = mu_day_res

            mu_intraday_res = sample_price_intraday(
                np.concatenate([mu_day, P_day]), P_intraday, t_idx + 1, Season
            )
            if isinstance(mu_intraday_res, tuple):
                mu_intraday, _ = mu_intraday_res
            else:
                mu_intraday = mu_intraday_res

            P_day_next = np.concatenate([mu_day, P_day[:-24]])
            P_intraday_next = np.concatenate([mu_intraday, P_intraday[:-96]])

            lk = 2
            VR_abc_neg = np.zeros((lk - 1, 3))
            VR_abc_pos = np.zeros((lk - 1, 3))

            if t_idx < T - 1:
                phi = np.hstack([P_day_sample_next, P_intraday_sample_next])
                Y = np.hstack([P_day_next, P_intraday_next])
                w = VRx_weights(phi, Y, weights_D_value[t_idx + 1, :])

                VRx = np.zeros((length_R, 3))
                for iR in range(length_R):
                    for jx in range(3):
                        VRx[iR, jx] = (Vt[iR, jx, :, t_idx + 1].reshape(1, N) @ w)[0]

                points = np.column_stack((R_vec, VRx[:, 1]))
                hull = ConvexHull(points)
                k = hull.vertices
                k = k[np.argsort(R_vec[k])]
                if len(k) > 1:
                    k = k[1:]
                lk = len(k)
                VR = VRx[k, :]
                R_k = R_vec[k]

                # Compute VR_abc_neg and VR_abc_pos only if we have at least two points in k
                if lk > 1:
                    VR_abc_neg = np.zeros((lk - 1, 3))
                    VR_abc_pos = np.zeros((lk - 1, 3))
                    for i in range(1, lk):
                        VR_abc_neg[i - 1, 1] = (VR[i, 1] - VR[i - 1, 1]) / (R_k[i] - R_k[i - 1])
                        VR_abc_neg[i - 1, 0] = VR[i, 1] - VR_abc_neg[i - 1, 1] * R_k[i]
                        VR_abc_neg[i - 1, 2] = -(VR[i - 1, 1] - VR[i - 1, 0]) / (x_vec[1] - x_vec[0])

                        VR_abc_pos[i - 1, 1] = (VR[i, 1] - VR[i - 1, 1]) / (R_k[i] - R_k[i - 1])
                        VR_abc_pos[i - 1, 0] = VR[i, 1] - VR_abc_pos[i - 1, 1] * R_k[i]
                        VR_abc_pos[i - 1, 2] = (VR[i - 1, 1] - VR[i - 1, 2]) / (x_vec[1] - x_vec[2])
                else:
                    # If lk <= 1, we cannot compute these slopes
                    VR_abc_neg = np.zeros((0, 3))
                    VR_abc_pos = np.zeros((0, 3))
                    
                # Add this part: Compute constraints only if lk > 1
                if lk > 1:
                    # lk-1 > 0, so VR_abc_neg and VR_abc_pos have rows
                    AV_neg = np.zeros((lk - 1, 12 * 96 + 24 + 1))
                    AV_neg[:, -1] = 1
                    AV_neg[:, 95] = -VR_abc_neg[:, 1]
                    AV_neg[:, (4 * 96) - 1] = -VR_abc_neg[:, 2]
                    bV_neg = VR_abc_neg[:, 0]

                    AV_pos = np.zeros((lk - 1, 12 * 96 + 24 + 1))
                    AV_pos[:, -1] = 1
                    AV_pos[:, 95] = -VR_abc_neg[:, 1]
                    AV_pos[:, (3 * 96) - 1] = -VR_abc_pos[:, 2]
                    bV_pos = VR_abc_pos[:, 0]
                else:
                    # If no segments, no additional constraints
                    AV_neg = np.zeros((0, 12 * 96 + 24 + 1))
                    bV_neg = np.zeros((0,))
                    AV_pos = np.zeros((0, 12 * 96 + 24 + 1))
                    bV_pos = np.zeros((0,))
                    
            else:
                # If t_idx >= T - 1, VR_abc_neg and VR_abc_pos are not computed
                VR_abc_neg = np.zeros((0, 3))
                VR_abc_pos = np.zeros((0, 3))


            # Solve MILP for each R, x0
            for iR in range(length_R):
                R_val = R_vec[iR]
                for ix_i, x0 in enumerate(x_vec):
                    f = np.zeros(96 * 12 + 24 + 1)
                    f[-1] = 1
                    # f(96+1:96*2)-=Delta_ti*mu_intraday
                    f[96:192] -= Delta_ti * mu_intraday
                    f[-25:-1] = -Delta_td * mu_day

                    q_pump_up = (
                        (np.abs(mu_intraday) / Q_mult - Q_fix) * t_ramp_pump_up / 2
                    )
                    q_pump_down = (
                        (np.abs(mu_intraday) * Q_mult + Q_fix) * t_ramp_pump_down / 2
                    )
                    q_turbine_up = (
                        (np.abs(mu_intraday) * Q_mult + Q_fix) * t_ramp_turbine_up / 2
                    )
                    q_turbine_down = (
                        (np.abs(mu_intraday) / Q_mult - Q_fix) * t_ramp_turbine_down / 2
                    )

                    f[192:288] -= c_grid_fee
                    f[384:480] += q_pump_up
                    f[480:576] -= q_pump_down
                    f[576:672] -= q_turbine_up
                    f[672:768] += q_turbine_down
                    f[960:1056] -= Q_start_pump
                    f[1056:1152] -= Q_start_turbine

                    # Construct constraints exactly as in MATLAB
                    # R_hq constraints
                    # A1:
                    A1 = np.concatenate(
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
                        ],
                        axis=1,
                    )
                    b1 = np.zeros((96, 1))
                    b1[0, 0] = -R_val

                    # Aufteilung in x_pump und x_turbine
                    Axh = np.zeros((96, 24))
                    for h in range(24):
                        Axh[h * 4 : (h + 1) * 4, h] = -1

                    A2 = np.concatenate(
                        [
                            np.zeros((96, 96)),
                            -np.eye(96),
                            np.eye(96),
                            -np.eye(96),
                            np.zeros((96, 96 * 8)),
                            Axh,
                            np.zeros((96, 1)),
                        ],
                        axis=1,
                    )
                    b2 = np.zeros((96, 1))

                    # Aufteilung in x_pump_up, x_turbine
                    A3 = np.concatenate(
                        [
                            np.zeros((96, 96 * 2)),
                            np.eye(96) - np.diag(np.ones(95), -1),
                            np.zeros((96, 96)),
                            -np.eye(96),
                            np.eye(96),
                            np.zeros((96, 96 * 6 + 24)),
                            np.zeros((96, 1)),
                        ],
                        axis=1,
                    )
                    b3 = np.zeros((96, 1))
                    b3[0, 0] = max(x0, 0)

                    # Aufteilung in x_turbine_up x_turbine_down
                    A4 = np.concatenate(
                        [
                            np.zeros((96, 96 * 3)),
                            np.eye(96) - np.diag(np.ones(95), -1),
                            np.zeros((96, 96 * 2)),
                            -np.eye(96),
                            np.eye(96),
                            np.zeros((96, 96 * 4 + 24)),
                            np.zeros((96, 1)),
                        ],
                        axis=1,
                    )
                    b4 = np.zeros((96, 1))
                    b4[0, 0] = max(-x0, 0)

                    Aeq = np.vstack([A1, A2, A3, A4])
                    beq = np.vstack([b1, b2, b3, b4])

                    # Nur Pumpen wenn Pumpe an
                    A1_2 = np.concatenate(
                        [
                            np.zeros((96, 96 * 2)),
                            -np.eye(96),
                            np.zeros((96, 96 * 5)),
                            x_min_pump * np.eye(96),
                            np.zeros((96, 96 * 3 + 24)),
                            np.zeros((96, 1)),
                        ],
                        axis=1,
                    )
                    A1_3 = np.concatenate(
                        [
                            np.zeros((96, 96 * 2)),
                            np.eye(96),
                            np.zeros((96, 96 * 5)),
                            -x_max_pump * np.eye(96),
                            np.zeros((96, 96 * 3 + 24)),
                            np.zeros((96, 1)),
                        ],
                        axis=1,
                    )
                    A1_4 = np.concatenate(
                        [
                            np.zeros((96, 96 * 3)),
                            -np.eye(96),
                            np.zeros((96, 96 * 5)),
                            x_min_turbine * np.eye(96),
                            np.zeros((96, 96 * 2 + 24)),
                            np.zeros((96, 1)),
                        ],
                        axis=1,
                    )
                    A1_5 = np.concatenate(
                        [
                            np.zeros((96, 96 * 3)),
                            np.eye(96),
                            np.zeros((96, 96 * 5)),
                            -x_max_turbine * np.eye(96),
                            np.zeros((96, 96 * 2 + 24)),
                            np.zeros((96, 1)),
                        ],
                        axis=1,
                    )

                    A1_all = np.vstack([A1_2, A1_3, A1_4, A1_5])
                    b1_all = np.zeros((96 * 4, 1))

                    # Wann wird Pumpe/turbine angestellt
                    A2_2 = np.concatenate(
                        [
                            np.zeros((96, 96 * 8)),
                            np.eye(96) - np.diag(np.ones(95), -1),
                            np.zeros((96, 96)),
                            -np.eye(96),
                            np.zeros((96, 96 + 24)),
                            np.zeros((96, 1)),
                        ],
                        axis=1,
                    )
                    b2_2 = np.zeros((96, 1))
                    b2_2[0, 0] = x0 > 0

                    A3_2 = np.concatenate(
                        [
                            np.zeros((96, 96 * 9)),
                            np.eye(96) - np.diag(np.ones(95), -1),
                            np.zeros((96, 96)),
                            -np.eye(96),
                            np.zeros((96, 24)),
                            np.zeros((96, 1)),
                        ],
                        axis=1,
                    )
                    b3_2 = np.zeros((96, 1))
                    b3_2[0, 0] = x0 < 0

                    # nur pumpe oder turbine anstellen
                    A4_2 = np.concatenate(
                        [
                            np.zeros((96, 96 * 8)),
                            np.eye(96),
                            np.eye(96),
                            np.zeros((96, 2 * 96 + 24)),
                            np.zeros((96, 1)),
                        ],
                        axis=1,
                    )
                    b4_2 = np.ones((96, 1))

                    # restriktionen für Wertfunktion
                    # AV_neg
                    # AV_neg(:,end)=1; AV_neg(:,96)=-VR_abc_neg(:,2); AV_neg(:,4*96)=-VR_abc_neg(:,3)
                    # We have lk-1 rows for AV_neg, from VR_abc_neg
                    if lk > 1:
                        AV_neg = np.zeros((lk - 1, 12 * 96 + 24 + 1))
                        AV_neg[:, -1] = 1
                        AV_neg[:, 95] = -VR_abc_neg[:, 1]
                        AV_neg[:, 4 * 96 - 1] = -VR_abc_neg[:, 2]
                        bV_neg = VR_abc_neg[:, 0]

                        AV_pos = np.zeros((lk - 1, 12 * 96 + 24 + 1))
                        AV_pos[:, -1] = 1
                        AV_pos[:, 95] = -VR_abc_neg[:, 1]
                        # AV_pos(:,3*96) = -VR_abc_neg(:,3); but we must use VR_abc_pos here:
                        AV_pos[:, 3 * 96 - 1] = -VR_abc_pos[:, 2]
                        bV_pos = VR_abc_pos[:, 0]
                    else:
                        # If lk=1, no AV_neg, AV_pos constraints
                        AV_neg = np.zeros((0, 12 * 96 + 24 + 1))
                        bV_neg = np.zeros((0,))
                        AV_pos = np.zeros((0, 12 * 96 + 24 + 1))
                        bV_pos = np.zeros((0,))

                    A = np.vstack([A1_all, A2_2, A3_2, A4_2, AV_neg, AV_pos])
                    b = np.vstack(
                        [
                            b1_all,
                            b2_2,
                            b3_2,
                            b4_2,
                            bV_neg.reshape(-1, 1),
                            bV_pos.reshape(-1, 1),
                        ]
                    )

                    lb = np.concatenate(
                        [
                            np.zeros(96),
                            -np.inf * np.ones(96),
                            np.zeros(96 * 10),
                            -x_max_turbine * np.ones(24),
                            [-np.inf],
                        ]
                    )
                    ub = np.concatenate(
                        [
                            Rmax * np.ones(96),
                            np.inf * np.ones(96 * 7),
                            np.ones(96 * 4),
                            x_max_pump * np.ones(24),
                            [np.inf],
                        ]
                    )

                    intcon = (
                        np.arange(8 * 96, 10 * 96) + 1
                    )  # +1 for MATLAB 1-based indexing

                    # Convert to MATLAB arrays
                    f_mat = matlab.double(f.tolist())
                    A_mat = matlab.double(A.tolist())
                    b_mat = matlab.double(b.flatten().tolist())
                    Aeq_mat = matlab.double(Aeq.tolist())
                    beq_mat = matlab.double(beq.flatten().tolist())
                    lb_mat = matlab.double(lb.tolist())
                    ub_mat = matlab.double(ub.tolist())
                    intcon_mat = matlab.double(intcon.tolist())

                    fneg_mat = matlab.double((-f).tolist())

                    sol = eng.intlinprog(
                        fneg_mat,
                        intcon_mat,
                        A_mat,
                        b_mat,
                        Aeq_mat,
                        beq_mat,
                        lb_mat,
                        ub_mat,
                        intlinprog_options,
                        nargout=4,
                    )

                    x_sol, fval, exitflag, output = sol
                    Vt[iR, ix_i, n_i, t_idx] = -fval

    # The MATLAB code does not show how EV is computed at the end.
    # We can return Vt and let user compute EV or set EV=0 as placeholder.

    # %%
    Vt

    # Compute EV if needed (the original code ends with EV?)
    # EV = 0  # or compute from Vt as required

    eng.quit()

    # %%

    return


# %%
