import numpy as np
from scipy.stats import multivariate_normal
import os
from scipy.io import loadmat

# Local imports
from sample_price_day import sample_price_day
from sample_price_intraday import sample_price_intraday


M=50
T=30
D=7
Season='Summer'

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

P_day_mat = loadmat(os.path.join('Data', f'P_day_{Season}.mat'))
P_intraday_mat = loadmat(os.path.join('Data', f'P_intraday_{Season}.mat'))

P_day_0 = P_day_mat['P_day_0'].flatten()
P_intraday_0 = P_intraday_mat['P_intraday_0'].flatten()
seed=2
np.random.seed(seed)

def generate_scenarios(N, T, D, P_day_0, P_intraday_0, Season, seed=None):
    if seed is not None:
        np.random.seed(seed)
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
    return sample_P_day_all, sample_P_intraday_all, Wt_day_mat, Wt_intra_mat

sample_P_day_all_fwd, sample_P_intraday_all_fwd, Wt_day_mat_fwd, Wt_intra_mat_fwd = generate_scenarios(M, T, D, P_day_0, P_intraday_0, Season, seed=seed)

R_0=0
x0_0=0
V = np.zeros((M,1))

R_path=np.zeros((M,96*T))
x_intraday_path=np.zeros((M,96*T))
P_day_path=np.zeros((M,96*T))
P_intraday_path=np.zeros((M,96*T))
x_pump_path=np.zeros((M,96*T))
x_turbine_path=np.zeros((M,96*T))
y_pump_path=np.zeros((M,96*T))
y_turbine_path=np.zeros((M,96*T))
z_pump_path=np.zeros((M,96*T))
z_turbine_path=np.zeros((M,96*T))

for m in range(M):
    print(f'Running for m={m}')
    R = R_0
    x0 = x0_0
    P_day = P_day_0[:24*D].copy()
    P_intraday = P_intraday_0[:96*D].copy()

    P_day_sim = P_day_0[:24*D].copy()
    P_intraday_sim = P_intraday_0[:96*D].copy()

    C=0
    for t_i in range(T):
        mu_day,_ = sample_price_day(P_day_sim, t_i, Season)
        mu_intraday,_ = sample_price_intraday(np.concatenate([mu_day,P_day_sim]), P_intraday_sim, t_i, Season)

        P_day_next = np.concatenate([mu_day, P_day[:-24].copy()])
        P_intraday_next = np.concatenate([mu_intraday, P_intraday[:-96].copy()])

        q_pump_up=(abs(mu_intraday)/Q_mult - Q_fix)*t_ramp_pump_up/2
        q_pump_down=(abs(mu_intraday)*Q_mult+Q_fix)*t_ramp_pump_down/2
        q_turbine_up=(abs(mu_intraday)*Q_mult+Q_fix)*t_ramp_turbine_up/2
        q_turbine_down=(abs(mu_intraday)/Q_mult - Q_fix)*t_ramp_turbine_down/2

        # Get day ahead initial state
        da_state = np.concatenate([[R], [x0], P_day, P_intraday])

        Wt_day=Wt_day_mat_fwd[m,t_i*24:(t_i+1)*24].copy()

        # Get day ahead action from corresponding policy model
        xday_opt = da_model.predict(da_state)

        # Get initial state for intraday
        id_state = np.concatenate([da_state, xday_opt, Wt_day])

        # Get intraday action from corresponding policy model
        x_opt2 = intraday_model.predict(id_state)

        # Extract results from x_opt2
        R_opt = x_opt2[:96].copy()
        xhq_opt = x_opt2[96:2*96].copy()

        Delta_pump_up = x_opt2[4*96:5*96].copy()
        Delta_pump_down = x_opt2[5*96:6*96].copy()
        Delta_turbine_up = x_opt2[6*96:7*96].copy()
        Delta_turbine_down = x_opt2[7*96:8*96].copy()

        x_pump = x_opt2[2*96:3*96].copy()
        x_turbine = x_opt2[3*96:4*96].copy()
        y_pump = x_opt2[8*96:9*96].copy()
        y_turbine = x_opt2[9*96:10*96].copy()
        z_pump = x_opt2[10*96:11*96].copy()
        z_turbine = x_opt2[11*96:12*96].copy()

        Wt_intraday = Wt_intra_mat_fwd[m, t_i*96:(t_i+1)*96].copy()
        P_intraday_path[m, t_i*96:(t_i+1)*96] = Wt_intraday

        # Update q_ ramps with realized intraday prices
        q_pump_up = (np.abs(Wt_intraday)/Q_mult - Q_fix)*t_ramp_pump_up/2
        q_pump_down = (np.abs(Wt_intraday)*Q_mult+Q_fix)*t_ramp_pump_down/2
        q_turbine_up = (np.abs(Wt_intraday)*Q_mult+Q_fix)*t_ramp_turbine_up/2
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

EV=np.mean(V)
print(EV)