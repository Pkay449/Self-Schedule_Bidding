# src/BADP/gen_offline_samples.py

import warnings

import matlab.engine
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

warnings.filterwarnings("ignore")

from src.config import SimulationParams
from src.utils.helpers import (
    badp_weights,
    build_and_solve_intlinprog,
    compute_weights,
    generate_scenarios,
    sample_price_day,
    sample_price_intraday,
)


def generate_offline_data(
    sim_params: SimulationParams,
    P_day_0: np.ndarray,
    P_intraday_0: np.ndarray,
    Vt: np.ndarray,
    P_day_state: np.ndarray,
    P_intra_state: np.ndarray,
):
    """
    Generate an offline dataset for training reinforcement learning models.

    This function simulates scenarios for day-ahead and intraday energy prices,
    solves a sequence of optimization problems to compute actions and rewards,
    and constructs datasets for both day-ahead and intraday stages. These datasets
    can then be used for offline policy training.

    Returns
    -------
    EV : float
        The expected value of cumulative cost/reward over all simulated scenarios.
    """
    # Number of scenarios and time steps
    N = 50  # Scenarios used for backward pass
    M = 50  # Scenarios for forward pass (training data)
    T = 3  # Optimization horizon

    # Start MATLAB engine
    eng = matlab.engine.start_matlab()

    weights_D_value = badp_weights(T)

    intlinprog_options = eng.optimoptions("intlinprog", "display", "off")

    np.random.seed(sim_params.seed + 1)
    (
        sample_P_day_all_fwd,
        sample_P_intraday_all_fwd,
        Wt_day_mat_fwd,
        Wt_intra_mat_fwd,
    ) = generate_scenarios(
        M,
        T,
        sim_params.D,
        P_day_0,
        P_intraday_0,
        sim_params.Season,
        seed=sim_params.seed + 1,
    )

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

    da_s = []  # R_Start , x_start, P_history = 1 + 1 + 24*7 + 96*7
    da_a = []
    da_r = []
    da_s_prime = []  # Da_s + x_day_opt + Wt+1_day

    id_s = []
    id_a = []
    id_r = []
    id_s_prime = []

    # Forward simulation across M scenarios
    for m in range(M):
        R = R_0  # Reset reservoir level for each scenario
        x0 = x0_0  # Reset intraday flow state
        P_day = P_day_0[
            : 24 * sim_params.D
        ].copy()  # take most recent 7 days DA up to the day
        P_intraday = P_intraday_0[
            : 96 * sim_params.D
        ].copy()  # take most recent 7 days ID up to the day

        # Simulated price paths
        P_day_sim = P_day_0[: 24 * sim_params.D].copy()
        P_intraday_sim = P_intraday_0[: 96 * sim_params.D].copy()

        C = 0
        for t_i in range(T):

            # Sample day-ahead and intraday prices
            mu_day, _ = sample_price_day(P_day_sim, t_i, sim_params.Season)
            mu_intraday, _ = sample_price_intraday(
                np.concatenate([mu_day, P_day_sim]),
                P_intraday_sim,
                t_i,
                sim_params.Season,
            )
            # Update price vectors for next step
            P_day_next = np.concatenate([mu_day, P_day[:-24].copy()])
            P_intraday_next = np.concatenate([mu_intraday, P_intraday[:-96].copy()])

            lk = 2
            VR_abc_neg = np.zeros((lk - 1, 3))
            VR_abc_pos = np.zeros((lk - 1, 3))

            # State for day-ahead optimization
            da_state = np.concatenate([[R], [x0], P_day, P_intraday])
            da_s.append(da_state)

            # Compute weights and piecewise linear approximations if not at final time step
            if t_i < T - 1 and np.any(Vt != 0):
                # Extract scenarios for next stage
                P_day_sample_next = P_day_state[:, t_i + 1, :].reshape(
                    N, sim_params.D * 24
                )
                P_intraday_sample_next = P_intra_state[:, t_i + 1, :].reshape(
                    N, sim_params.D * 24 * 4
                )

                phi = np.hstack((P_day_sample_next, P_intraday_sample_next))
                Y = np.hstack((P_day_next, P_intraday_next))

                # Compute weights for next stage
                weights = compute_weights(eng, phi, Y, weights_D_value[int(t_i + 1), :])

                # Value function approximations
                VRx = np.zeros((sim_params.length_R, 3))
                for i in range(sim_params.length_R):
                    for j in range(3):
                        VRx[i, j] = Vt[i, j, :, t_i + 1].dot(weights)

                # Compute piecewise approximations using convex hull
                hull_input = np.column_stack([sim_params.R_vec.T, VRx[:, 1]])
                hull = ConvexHull(hull_input)
                k = hull.vertices
                k = np.sort(k)
                k = k[1:]  # Remove the first vertex
                lk = len(k)

                VR = VRx[k, :]
                R_k = sim_params.R_vec[k]

                if lk > 1:
                    VR_abc_neg = np.zeros((lk - 1, 3))
                    VR_abc_pos = np.zeros((lk - 1, 3))
                    for i in range(1, lk):
                        VR_abc_neg[i - 1, 1] = (VR[i, 1] - VR[i - 1, 1]) / (
                            R_k[i] - R_k[i - 1]
                        )
                        VR_abc_neg[i - 1, 0] = VR[i, 1] - VR_abc_neg[i - 1, 1] * R_k[i]
                        VR_abc_neg[i - 1, 2] = -(VR[i - 1, 1] - VR[i - 1, 0]) / (
                            sim_params.x_vec[1] - sim_params.x_vec[0]
                        )

                    for i in range(1, lk):
                        VR_abc_pos[i - 1, 1] = (VR[i, 1] - VR[i - 1, 1]) / (
                            R_k[i] - R_k[i - 1]
                        )
                        VR_abc_pos[i - 1, 0] = VR[i, 1] - VR_abc_pos[i - 1, 1] * R_k[i]
                        VR_abc_pos[i - 1, 2] = (VR[i - 1, 1] - VR[i - 1, 2]) / (
                            sim_params.x_vec[1] - sim_params.x_vec[2]
                        )
                else:
                    VR_abc_neg = np.zeros((0, 3))
                    VR_abc_pos = np.zeros((0, 3))

            # Now build the MILP for the forward pass
            # First MILP to get xday_opt
            # Build f
            f = np.zeros(96 * 12 + 24 + 1)
            f[-1] = 1
            f[96:192] -= sim_params.Delta_ti * mu_intraday
            f[-25:-1] = -sim_params.Delta_td * mu_day

            q_pump_up = (
                (abs(mu_intraday) / sim_params.Q_mult - sim_params.Q_fix)
                * sim_params.t_ramp_pump_up
                / 2
            )
            q_pump_down = (
                (abs(mu_intraday) * sim_params.Q_mult + sim_params.Q_fix)
                * sim_params.t_ramp_pump_down
                / 2
            )
            q_turbine_up = (
                (abs(mu_intraday) * sim_params.Q_mult + sim_params.Q_fix)
                * sim_params.t_ramp_turbine_up
                / 2
            )
            q_turbine_down = (
                (abs(mu_intraday) / sim_params.Q_mult - sim_params.Q_fix)
                * sim_params.t_ramp_turbine_down
                / 2
            )

            f[96 * 2 : 96 * 3] -= sim_params.c_grid_fee
            f[96 * 4 : 96 * 5] += q_pump_up
            f[96 * 5 : 96 * 6] -= q_pump_down
            f[96 * 6 : 96 * 7] -= q_turbine_up
            f[96 * 7 : 96 * 8] += q_turbine_down
            f[96 * 10 : 96 * 11] -= sim_params.Q_start_pump
            f[96 * 11 : 96 * 12] -= sim_params.Q_start_turbine

            # Build constraints for first MILP (day-ahead)
            A1 = np.hstack(
                [
                    -np.eye(96) + np.diag(np.ones(95), -1),
                    np.zeros((96, 96)),
                    sim_params.Delta_ti * sim_params.beta_pump * np.eye(96),
                    -sim_params.Delta_ti / sim_params.beta_turbine * np.eye(96),
                    -sim_params.beta_pump * sim_params.c_pump_up * np.eye(96),
                    sim_params.beta_pump * sim_params.c_pump_down * np.eye(96),
                    sim_params.c_turbine_up / sim_params.beta_turbine * np.eye(96),
                    -sim_params.c_turbine_down / sim_params.beta_turbine * np.eye(96),
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
                            sim_params.x_min_pump * np.eye(96),
                            np.zeros((96, 96 * 3 + 24)),
                            np.zeros((96, 1)),
                        ]
                    ),
                    np.hstack(
                        [
                            np.zeros((96, 96 * 2)),
                            np.eye(96),
                            np.zeros((96, 96 * 5)),
                            -sim_params.x_max_pump * np.eye(96),
                            np.zeros((96, 96 * 3 + 24)),
                            np.zeros((96, 1)),
                        ]
                    ),
                    np.hstack(
                        [
                            np.zeros((96, 96 * 3)),
                            -np.eye(96),
                            np.zeros((96, 96 * 5)),
                            sim_params.x_min_turbine * np.eye(96),
                            np.zeros((96, 96 * 2 + 24)),
                            np.zeros((96, 1)),
                        ]
                    ),
                    np.hstack(
                        [
                            np.zeros((96, 96 * 3)),
                            np.eye(96),
                            np.zeros((96, 96 * 5)),
                            -sim_params.x_max_turbine * np.eye(96),
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
                    -sim_params.x_max_turbine * np.ones(24),
                    np.full(1, -np.inf),
                ]
            )
            ub = np.concatenate(
                [
                    sim_params.Rmax * np.ones(96),
                    np.inf * np.ones(96 * 7),
                    np.ones(96 * 4),
                    sim_params.x_max_pump * np.ones(24),
                    np.full(1, np.inf),
                ]
            )

            intcon = np.arange(8 * 96, 96 * 10)
            # Solve first MILP for day-ahead
            x_opt, fval = build_and_solve_intlinprog(
                eng, f, A, b, Aeq, beq, lb, ub, intcon, intlinprog_options
            )

            xday_opt = x_opt[-25:-1].copy()

            da_a.append(xday_opt)

            Wt_day = Wt_day_mat_fwd[
                m, t_i * 24 : (t_i + 1) * 24
            ].copy()  # THE NEXT DA day Prices
            day_path = np.tile(Wt_day, (4, 1))
            P_day_path[m, t_i * 96 : (t_i + 1) * 96] = day_path.flatten()

            da_r.append(-sim_params.Delta_td * np.dot(Wt_day, xday_opt))

            mu_intraday, _ = sample_price_intraday(
                np.concatenate([Wt_day, P_day_sim]),
                P_intraday_sim,
                t_i,
                sim_params.Season,
            )

            P_day_next = np.concatenate(
                [Wt_day, P_day[:-24].copy()]
            )  # Remove oldest day from DA price series
            da_next_state = np.concatenate([da_state, xday_opt, Wt_day])

            da_s_prime.append(da_next_state)
            id_s.append(da_next_state)
            P_intraday_next = np.concatenate(
                [mu_intraday, P_intraday[:-96].copy()]
            )  # Remove oldest day from ID price series

            # Now solve the second MILP (intraday) with xday_opt as bounds
            # Build f again for intraday stage
            f = np.zeros(96 * 12 + 24 + 1)
            f[-1] = 1
            f[96:192] -= sim_params.Delta_ti * mu_intraday
            f[-25:-1] = -sim_params.Delta_td * mu_day
            q_pump_up = (
                (abs(mu_intraday) / sim_params.Q_mult - sim_params.Q_fix)
                * sim_params.t_ramp_pump_up
                / 2
            )
            q_pump_down = (
                (abs(mu_intraday) * sim_params.Q_mult + sim_params.Q_fix)
                * sim_params.t_ramp_pump_down
                / 2
            )
            q_turbine_up = (
                (abs(mu_intraday) * sim_params.Q_mult + sim_params.Q_fix)
                * sim_params.t_ramp_turbine_up
                / 2
            )
            q_turbine_down = (
                (abs(mu_intraday) / sim_params.Q_mult - sim_params.Q_fix)
                * sim_params.t_ramp_turbine_down
                / 2
            )
            f[96 * 2 : 96 * 3] -= sim_params.c_grid_fee
            f[96 * 4 : 96 * 5] += q_pump_up
            f[96 * 5 : 96 * 6] -= q_pump_down
            f[96 * 6 : 96 * 7] -= q_turbine_up
            f[96 * 7 : 96 * 8] += q_turbine_down
            f[96 * 10 : 96 * 11] -= sim_params.Q_start_pump
            f[96 * 11 : 96 * 12] -= sim_params.Q_start_turbine

            # R_hq
            A1 = [
                -np.eye(96) + np.diag(np.ones(95), -1),
                np.zeros((96, 96)),
                sim_params.Delta_ti * sim_params.beta_pump * np.eye(96),
                -sim_params.Delta_ti / sim_params.beta_turbine * np.eye(96),
                -sim_params.beta_pump * sim_params.c_pump_up * np.eye(96),
                sim_params.beta_pump * sim_params.c_pump_down * np.eye(96),
                sim_params.c_turbine_up / sim_params.beta_turbine * np.eye(96),
                -sim_params.c_turbine_down / sim_params.beta_turbine * np.eye(96),
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
                sim_params.x_min_pump * np.eye(96),
                np.zeros((96, 96 * 3 + 24)),
                np.zeros((96, 1)),
                np.zeros(
                    (
                        96,
                        96,
                    )
                ),
            ]

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
                    sim_params.Rmax * np.ones(96),
                    np.inf * np.ones(96 * 7),
                    np.ones(96 * 4),
                    xday_opt,
                    np.full(1, np.inf),
                ]
            )

            x_opt2, fval2 = build_and_solve_intlinprog(
                eng, f, A, b, Aeq, beq, lb, ub, intcon, intlinprog_options
            )

            action_id = x_opt2[:-25]

            id_a.append(action_id)

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
            q_pump_up = (
                (np.abs(Wt_intraday) / sim_params.Q_mult - sim_params.Q_fix)
                * sim_params.t_ramp_pump_up
                / 2
            )
            q_pump_down = (
                (np.abs(Wt_intraday) * sim_params.Q_mult + sim_params.Q_fix)
                * sim_params.t_ramp_pump_down
                / 2
            )
            q_turbine_up = (
                (np.abs(Wt_intraday) * sim_params.Q_mult + sim_params.Q_fix)
                * sim_params.t_ramp_turbine_up
                / 2
            )
            q_turbine_down = (
                (np.abs(Wt_intraday) / sim_params.Q_mult - sim_params.Q_fix)
                * sim_params.t_ramp_turbine_down
                / 2
            )

            # Update R, x0, P_day, P_intraday, P_day_sim, P_intraday_sim
            R = R_opt[-1].copy()
            x0 = x_pump[-1] - x_turbine[-1]
            P_day = np.concatenate([Wt_day, P_day[:-24].copy()])
            P_intraday = np.concatenate([Wt_intraday, P_intraday[:-96].copy()])
            P_day_sim = np.concatenate([Wt_day, P_day_sim[:-24].copy()])
            P_intraday_sim = np.concatenate([Wt_intraday, P_intraday_sim[:-96].copy()])

            id_r.append(
                -np.sum(x_pump) * sim_params.c_grid_fee
                - sim_params.Delta_ti * np.dot(Wt_intraday, xhq_opt)
                + np.dot(q_pump_up, Delta_pump_up)
                - np.dot(q_pump_down, Delta_pump_down)
                - np.dot(q_turbine_up, Delta_turbine_up)
                + np.dot(q_turbine_down, Delta_turbine_down)
                - np.sum(z_pump) * sim_params.Q_start_pump
                - np.sum(z_turbine) * sim_params.Q_start_turbine
            )

            next_state = np.concatenate([[R], [x0], P_day, P_intraday])
            id_s_prime.append(next_state)

            # Update C
            C = (
                C
                - sim_params.Delta_td * np.dot(Wt_day, xday_opt)
                - np.sum(x_pump) * sim_params.c_grid_fee
                - sim_params.Delta_ti * np.dot(Wt_intraday, xhq_opt)
                + np.dot(q_pump_up, Delta_pump_up)
                - np.dot(q_pump_down, Delta_pump_down)
                - np.dot(q_turbine_up, Delta_turbine_up)
                + np.dot(q_turbine_down, Delta_turbine_down)
                - np.sum(z_pump) * sim_params.Q_start_pump
                - np.sum(z_turbine) * sim_params.Q_start_turbine
            )

        V[m] = C

    EV = np.mean(V)

    # Save offline dataset as pickle

    df_da = pd.DataFrame(
        {"state": da_s, "action": da_a, "reward": da_r, "next_state": da_s_prime}
    )

    # df_da.to_pickle("Data/offline_dataset_day_ahead.pkl")

    df_id = pd.DataFrame(
        {"state": id_s, "action": id_a, "reward": id_r, "next_state": id_s_prime}
    )

    # df_id.to_pickle("Data/offline_dataset_intraday.pkl")

    return EV, df_da, df_id
