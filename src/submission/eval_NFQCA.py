#%%
import numpy as np
from scipy.stats import multivariate_normal
import os
from scipy.io import loadmat
# Local imports
from sample_price_day import sample_price_day
from sample_price_intraday import sample_price_intraday
from helper_NFQCA import generate_scenarios

def eval_learned_policy(policy_id_model, policy_da_model, policy_id_params, policy_da_params, M=10, T=30, D=7, Season="Summer", seed=2):
    """
    Evaluate a learned policy using forward simulation of day-ahead and intraday decisions.

    Parameters
    ----------
    policy_id_model : object
        Intraday policy model with an `apply` method.
    policy_da_model : object
        Day-ahead policy model with an `apply` method.
    policy_id_params : any
        Parameters for the intraday policy model.
    policy_da_params : any
        Parameters for the day-ahead policy model.
    M : int, optional
        Number of scenarios to simulate. Default is 10.
    T : int, optional
        Number of time steps to simulate. Default is 30.
    D : int, optional
        Number of days in the forecast. Default is 7.
    Season : str, optional
        Season to select the appropriate data files (e.g., "Summer"). Default is "Summer".
    seed : int, optional
        Random seed for reproducibility. Default is 2.

    Returns
    -------
    float
        Expected value (EV) of the policy based on the simulation.
    """
    Q_mult=1.2
    Q_fix=3
    Q_start_pump=15
    Q_start_turbine=15
    c_grid_fee=5/4
    Delta_ti=0.25
    Delta_td=1.0
    t_ramp_pump_up=2/60
    t_ramp_pump_down=2/60
    t_ramp_turbine_up=2/60
    t_ramp_turbine_down=2/60

    np.random.seed(seed)
    try:
        P_day_mat = loadmat(os.path.join('Data', f'P_day_{Season}.mat'))
        P_intraday_mat = loadmat(os.path.join('Data', f'P_intraday_{Season}.mat'))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required data file is missing: {e.filename}")

    P_day_0 = P_day_mat['P_day_0'].flatten()
    P_intraday_0 = P_intraday_mat['P_intraday_0'].flatten()

    _, _, Wt_day_mat_fwd, Wt_intra_mat_fwd = generate_scenarios(M, T, D, P_day_0, P_intraday_0, Season, seed=seed)

    R_0, x0_0 = 0, 0
    V = np.zeros((M,1))
    P_intraday_path=np.zeros((M,96*T))

    for m in range(M):
        print(f'Running for m={m}')
        R, x0= R_0, x0_0
        P_day = P_day_0[: 24 * D].copy()
        P_intraday = P_intraday_0[: 96 * D].copy()
        P_day_sim = P_day_0[: 24 * D].copy()
        P_intraday_sim = P_intraday_0[: 96 * D].copy()
        C=0
        for t_i in range(T):
            mu_day,_ = sample_price_day(P_day_sim, t_i, Season)
            mu_intraday,_ = sample_price_intraday(np.concatenate([mu_day,P_day_sim]), P_intraday_sim, t_i, Season)

            # Update day-ahead and intraday prices
            P_day_next = np.concatenate([mu_day, P_day[:-24].copy()])
            P_intraday_next = np.concatenate([mu_intraday, P_intraday[:-96].copy()])
            # Compute ramping penalties
            q_pump_up=(abs(mu_intraday)/Q_mult - Q_fix)*t_ramp_pump_up/2
            q_pump_down=(abs(mu_intraday)*Q_mult+Q_fix)*t_ramp_pump_down/2
            q_turbine_up=(abs(mu_intraday)*Q_mult+Q_fix)*t_ramp_turbine_up/2
            q_turbine_down=(abs(mu_intraday)/Q_mult - Q_fix)*t_ramp_turbine_down/2

            # Get day ahead initial state
            da_state = np.concatenate([[R], [x0], P_day, P_intraday])
            Wt_day=Wt_day_mat_fwd[m,t_i*24:(t_i + 1) * 24].copy()

            # Get day ahead action from corresponding policy model
            xday_opt = policy_da_model.apply(policy_da_params, da_state)
            id_state = np.concatenate([da_state, xday_opt, Wt_day])
            xid_opt = policy_id_model.apply(policy_id_params, id_state)

            # Extract results from x_opt2
            R_opt = xid_opt[: 96].copy()
            xhq_opt = xid_opt[96:2 * 96].copy()

            Delta_pump_up = xid_opt[4 * 96:5 * 96].copy()
            Delta_pump_down = xid_opt[5 * 96:6 * 96].copy()
            Delta_turbine_up = xid_opt[6 * 96:7 * 96].copy()
            Delta_turbine_down = xid_opt[7 * 96:8 * 96].copy()

            x_pump = xid_opt[2 * 96:3 * 96].copy()
            x_turbine = xid_opt[3 * 96:4 * 96].copy()
            y_pump = xid_opt[8 * 96:9 * 96].copy()
            y_turbine = xid_opt[9 * 96:10 * 96].copy()
            z_pump = xid_opt[10 * 96:11 * 96].copy()
            z_turbine = xid_opt[11 * 96:12 * 96].copy()

            Wt_intraday = Wt_intra_mat_fwd[m, t_i * 96:(t_i+1) * 96].copy()
            P_intraday_path[m, t_i * 96:(t_i+1) * 96] = Wt_intraday

            # Update q_ ramps with realized intraday prices
            q_pump_up = (np.abs(Wt_intraday)/Q_mult - Q_fix)*t_ramp_pump_up/2
            q_pump_down = (np.abs(Wt_intraday) * Q_mult+Q_fix)*t_ramp_pump_down/2
            q_turbine_up = (np.abs(Wt_intraday) * Q_mult+Q_fix)*t_ramp_turbine_up/2
            q_turbine_down = (np.abs(Wt_intraday)/Q_mult - Q_fix)*t_ramp_turbine_down/2

            # Update R, x0, P_day, P_intraday, P_day_sim, P_intraday_sim
            R = R_opt[-1].copy()
            x0 = x_pump[-1] - x_turbine[-1]
            P_day = np.concatenate([Wt_day, P_day[:-24].copy()])
            P_intraday = np.concatenate([Wt_intraday, P_intraday[:-96].copy()])
            P_day_sim = np.concatenate([Wt_day, P_day_sim[:-24].copy()])
            P_intraday_sim = np.concatenate([Wt_intraday, P_intraday_sim[:-96].copy()])

            # Update C
            C = C - Delta_td*np.dot(Wt_day, xday_opt) - np.sum(x_pump)*c_grid_fee \
                - Delta_ti*np.dot(Wt_intraday, xhq_opt) \
                + np.dot(q_pump_up, Delta_pump_up) - np.dot(q_pump_down, Delta_pump_down) \
                - np.dot(q_turbine_up, Delta_turbine_up) + np.dot(q_turbine_down, Delta_turbine_down) \
                - np.sum(z_pump)*Q_start_pump - np.sum(z_turbine)*Q_start_turbine

        V[m] = C

    EV = np.mean(V)
    print(EV)