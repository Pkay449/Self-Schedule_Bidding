# src/Sequential_NFQCA/utils/evaluation.py

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from src.utils.helpers import sample_price_day, sample_price_intraday


def eval_learned_policy(
    policy_id_model, policy_da_model, policy_id_params, policy_da_params, save_path
):
    """
    Evaluate the performance of the learned policies for intraday and day-ahead energy trading.

    This function simulates the energy trading environment using the learned policies and
    evaluates their performance over multiple time steps. It calculates cumulative rewards
    and tracks key operational metrics.

    Parameters
    ----------
    policy_id_model : callable
        Intraday policy model, which takes intraday state and parameters as input and outputs actions.
    policy_da_model : callable
        Day-ahead policy model, which takes day-ahead state and parameters as input and outputs actions.
    policy_id_params : dict or array
        Parameters of the intraday policy model.
    policy_da_params : dict or array
        Parameters of the day-ahead policy model.

    Returns
    -------
    None
        Saves evaluation results, including paths, cumulative rewards, and environment trackers, to disk.
    """
    M = 1  # Number of scenarios to simulate
    T = 30  # Optimization horizon (time steps)
    D = 7  # Days considered for initial price series
    Season = "Summer"  # Season for the test dataset

    Q_mult = 1.2
    Q_fix = 3
    Q_start_pump = 15
    Q_start_turbine = 15

    c_grid_fee = 5 / 4
    Delta_ti = 0.25
    Delta_td = 1.0

    t_ramp_pump_up = 2 / 60
    t_ramp_pump_down = 2 / 60
    t_ramp_turbine_up = 2 / 60
    t_ramp_turbine_down = 2 / 60

    seed = 2
    np.random.seed(seed)

    P_day_mat = loadmat(os.path.join("Data", f"P_day_ahead_test_all.mat"))
    P_intraday_mat = loadmat(os.path.join("Data", f"P_intraday_test_all.mat"))

    P_day_0 = P_day_mat["P_day_0"].flatten()
    P_intraday_0 = P_intraday_mat["P_intraday_0"].flatten()

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

    # Enviroment trackers
    storage_track = []
    storage_track.append(R_0)

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

            # Compute ramping costs for intraday adjustments
            q_pump_up = (abs(mu_intraday) / Q_mult - Q_fix) * t_ramp_pump_up / 2
            q_pump_down = (abs(mu_intraday) * Q_mult + Q_fix) * t_ramp_pump_down / 2
            q_turbine_up = (abs(mu_intraday) * Q_mult + Q_fix) * t_ramp_turbine_up / 2
            q_turbine_down = (
                (abs(mu_intraday) / Q_mult - Q_fix) * t_ramp_turbine_down / 2
            )

            # Get day ahead initial state
            da_state = np.concatenate([[R], [x0], P_day, P_intraday])

            Wt_day = P_day_0[t_i * 24 : (t_i + 1) * 24].copy()
            day_path = np.tile(Wt_day, (4, 1))
            P_day_path[m, t_i * 96 : (t_i + 1) * 96] = day_path.flatten()

            # Get day ahead action from corresponding policy model
            xday_opt = policy_da_model.apply(policy_da_params, da_state)

            # Get initial state for intraday
            id_state = np.concatenate([da_state, xday_opt, Wt_day])

            # Get intraday action from corresponding policy model
            xid_opt = policy_id_model.apply(policy_id_params, id_state)

            # Extract results from x_opt2
            R_opt = xid_opt[:96].copy()
            xhq_opt = xid_opt[96 : 2 * 96].copy()

            Delta_pump_up = xid_opt[4 * 96 : 5 * 96].copy()
            Delta_pump_down = xid_opt[5 * 96 : 6 * 96].copy()
            Delta_turbine_up = xid_opt[6 * 96 : 7 * 96].copy()
            Delta_turbine_down = xid_opt[7 * 96 : 8 * 96].copy()

            x_pump = xid_opt[2 * 96 : 3 * 96].copy()
            x_turbine = xid_opt[3 * 96 : 4 * 96].copy()
            y_pump = xid_opt[8 * 96 : 9 * 96].copy()
            y_turbine = xid_opt[9 * 96 : 10 * 96].copy()
            z_pump = xid_opt[10 * 96 : 11 * 96].copy()
            z_turbine = xid_opt[11 * 96 : 12 * 96].copy()

            # Update paths
            R_path[m, t_i * 96 : (t_i + 1) * 96] = R_opt
            x_intraday_path[m, t_i * 96 : (t_i + 1) * 96] = xhq_opt
            x_pump_path[m, t_i * 96 : (t_i + 1) * 96] = x_pump
            x_turbine_path[m, t_i * 96 : (t_i + 1) * 96] = x_turbine
            y_pump_path[m, t_i * 96 : (t_i + 1) * 96] = y_pump
            y_turbine_path[m, t_i * 96 : (t_i + 1) * 96] = y_turbine
            z_pump_path[m, t_i * 96 : (t_i + 1) * 96] = z_pump
            z_turbine_path[m, t_i * 96 : (t_i + 1) * 96] = z_turbine

            Wt_intraday = P_intraday_0[t_i * 96 : (t_i + 1) * 96].copy()
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

            # Update R, x0, P_day, P_intraday, P_day_sim, P_intraday_sim
            R = R_opt[-1].copy()
            x0 = x_pump[-1] - x_turbine[-1]
            P_day = np.concatenate([Wt_day, P_day[:-24].copy()])
            P_intraday = np.concatenate([Wt_intraday, P_intraday[:-96].copy()])
            P_day_sim = np.concatenate([Wt_day, P_day_sim[:-24].copy()])
            P_intraday_sim = np.concatenate([Wt_intraday, P_intraday_sim[:-96].copy()])

            # Update C
            C = (
                C
                - Delta_td * np.dot(Wt_day, xday_opt)
                - np.sum(x_pump) * c_grid_fee
                - Delta_ti * np.dot(Wt_intraday, xhq_opt)
                + np.dot(q_pump_up, Delta_pump_up)
                - np.dot(q_pump_down, Delta_pump_down)
                - np.dot(q_turbine_up, Delta_turbine_up)
                + np.dot(q_turbine_down, Delta_turbine_down)
                - np.sum(z_pump) * Q_start_pump
                - np.sum(z_turbine) * Q_start_turbine
            )

            # UPDATE TRACKERS
            storage_track.append(R)

        V[m] = C

    EV = np.mean(V)
    print(EV)

    # print backtest statistics :
    print("Backtest Statistics:")
    print("Mean Value: ", np.mean(V))
    print("Standard Deviation: ", np.std(V))
    print("Total Reward: ", np.sum(V))

    # save trackers
    # storage_track
    np.save(os.path.join(save_path, "BACKTEST_storage_track.npy"), storage_track)

    # Save paths
    np.save(os.path.join(save_path, "BACKTEST_R_path.npy"), R_path)
    np.save(os.path.join(save_path, "BACKTEST_x_intraday_path.npy"), x_intraday_path)
    np.save(os.path.join(save_path, "BACKTEST_P_day_path.npy"), P_day_path)
    np.save(os.path.join(save_path, "BACKTEST_P_intraday_path.npy"), P_intraday_path)
    np.save(os.path.join(save_path, "BACKTEST_x_pump_path.npy"), x_pump_path)
    np.save(os.path.join(save_path, "BACKTEST_x_turbine_path.npy"), x_turbine_path)
    np.save(os.path.join(save_path, "BACKTEST_y_pump_path.npy"), y_pump_path)
    np.save(os.path.join(save_path, "BACKTEST_y_turbine_path.npy"), y_turbine_path)
    np.save(os.path.join(save_path, "BACKTEST_z_pump_path.npy"), z_pump_path)
    np.save(os.path.join(save_path, "BACKTEST_z_turbine_path.npy"), z_turbine_path)


def plot_backtest_results(results_path):
    R_path = np.load(os.path.join(results_path, "BACKTEST_R_path.npy")).ravel()
    x_intraday_path = np.load(
        os.path.join(results_path, "BACKTEST_x_intraday_path.npy")
    ).ravel()
    P_day_path = np.load(os.path.join(results_path, "BACKTEST_P_day_path.npy")).ravel()
    P_intraday_path = np.load(
        os.path.join(results_path, "BACKTEST_P_intraday_path.npy")
    ).ravel()
    x_pump_path = np.load(
        os.path.join(results_path, "BACKTEST_x_pump_path.npy")
    ).ravel()
    x_turbine_path = np.load(
        os.path.join(results_path, "BACKTEST_x_turbine_path.npy")
    ).ravel()
    y_pump_path = np.load(
        os.path.join(results_path, "BACKTEST_y_pump_path.npy")
    ).ravel()
    y_turbine_path = np.load(
        os.path.join(results_path, "BACKTEST_y_turbine_path.npy")
    ).ravel()
    z_pump_path = np.load(
        os.path.join(results_path, "BACKTEST_z_pump_path.npy")
    ).ravel()

    # Create a figure with multiple subplots
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle("Backtest Data Plots", fontsize=16)

    # Plot each array in a subplot
    data = {
        "R Path": R_path,
        "x Intraday Path": x_intraday_path,
        "P Day Path": P_day_path,
        "P Intraday Path": P_intraday_path,
        "x Pump Path": x_pump_path,
        "x Turbine Path": x_turbine_path,
        "y Pump Path": y_pump_path,
        "y Turbine Path": y_turbine_path,
        "z Pump Path": z_pump_path,
    }

    # Iterate through data and subplots
    for ax, (title, array) in zip(axs.flat, data.items()):
        ax.plot(array)
        ax.set_title(title)
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Values")
        ax.grid(True)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(results_path, "backtest_plots.png"))
    plt.show()
