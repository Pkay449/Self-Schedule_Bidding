import numpy as np
import pandas as pd
from scipy.io import loadmat
import jax
import jax.numpy as jnp
import pickle as pkl
import warnings
import matlab.engine
import os

from sample_price_day import sample_price_day
from sample_price_intraday import sample_price_intraday
from VRx_weights_pk import VRx_weights
from badp_weights_r import badp_weights
from helper import generate_scenarios  # Assuming you still have this helper

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load previously trained policy and Q-function parameters
with open('Results/q_params.pkl','rb') as f:
    q_params = pkl.load(f)
with open('Results/policy_params.pkl','rb') as f:
    policy_params = pkl.load(f)

# Define the Q and policy functions as used in training
def mlp(params, x):
    for W,b in params[:-1]:
        x = jnp.tanh(jnp.dot(x,W)+b)
    W,b = params[-1]
    return jnp.dot(x,W)+b

def q_function(q_params, s, a):
    sa = jnp.concatenate([s,a], axis=-1)
    return mlp(q_params, sa)

def policy(policy_params, s):
    x = s
    for W,b in policy_params[:-1]:
        x = jnp.tanh(jnp.dot(x,W)+b)
    W,b = policy_params[-1]
    a = jnp.dot(x,W)+b
    return a

# =====================
# Parameters (same as before)
# =====================
length_R = 5
N = 50
T = 3
M = 10
seed = 2
Season = "Summer"
D = 7
Rmax = 100
np.random.seed(seed)

t_ramp_pump_up=2/60
t_ramp_pump_down=2/60
t_ramp_turbine_up=2/60
t_ramp_turbine_down=2/60

c_grid_fee=5/4
Delta_ti=0.25
Delta_td=1.0

Q_mult=1.2
Q_fix=3
Q_start_pump=15
Q_start_turbine=15

beta_pump=0.9
beta_turbine=0.9

x_max_pump=10
x_min_pump=5
x_max_turbine=10
x_min_turbine=5

R_vec=np.linspace(0,Rmax,length_R)
x_vec=np.array([-x_max_turbine,0,x_max_pump])

c_pump_up=t_ramp_pump_up/2
c_pump_down=t_ramp_pump_down/2
c_turbine_up=t_ramp_turbine_up/2
c_turbine_down=t_ramp_turbine_down/2

# Load base data
P_day_mat = loadmat(os.path.join("Data", f"P_day_{Season}.mat"))
P_intraday_mat = loadmat(os.path.join("Data", f"P_intraday_{Season}.mat"))

P_day_0 = P_day_mat["P_day_0"].flatten()
P_intraday_0 = P_intraday_mat["P_intraday_0"].flatten()

# No MILP solver needed now, no VRx_weights for decisions since we rely on learned policy
# We'll just simulate forward and pick actions from the policy.

def evaluate_learned_policy():
    # Generate forward scenarios
    np.random.seed(seed+1)
    sample_P_day_all_fwd, sample_P_intraday_all_fwd, Wt_day_mat_fwd, Wt_intra_mat_fwd = generate_scenarios(M, T, D, P_day_0, P_intraday_0, Season, seed=seed+1)

    R_0=0
    x0_0=0
    V = np.zeros((M,1))

    for m in range(M):
        R = R_0
        x0 = x0_0
        P_day = P_day_0[:24*D].copy()
        P_intraday = P_intraday_0[:96*D].copy()

        P_day_sim = P_day_0[:24*D].copy()
        P_intraday_sim = P_intraday_0[:96*D].copy()

        C=0
        for t_i in range(T):
            # Compute mu_day, mu_intraday as before
            mu_day,_=sample_price_day(P_day_sim, t_i, Season)
            mu_intraday,_=sample_price_intraday(jnp.concatenate([mu_day, P_day_sim]), P_intraday_sim, t_i, Season)

            # Construct state for the policy
            # state s = [x0, R, P_day, P_intraday]
            # Convert to jax arrays
            s = jnp.concatenate([jnp.array([x0]), jnp.array([R]), jnp.array(P_day), jnp.array(P_intraday)])

            # Get action from policy
            # action dimension: day-ahead (24) + intraday(96)? Let's assume action shape matches original code
            a = policy(policy_params, s)

            # Split a into day-ahead portion and intraday portion if needed
            xday_opt = a[:24]
            xhq_opt = a[24:24+96]  # Adjust if needed

            # Simulate environment forward
            Wt_day = Wt_day_mat_fwd[m, t_i*24:(t_i+1)*24].copy()
            Wt_intraday = Wt_intra_mat_fwd[m, t_i*96:(t_i+1)*96].copy()

            # Update next state
            # After applying action, we must update R,x0,P_day,P_intraday
            # Based on original code logic
            q_pump_up = (np.abs(Wt_intraday)/Q_mult - Q_fix)*t_ramp_pump_up/2
            q_pump_down = (np.abs(Wt_intraday)*Q_mult+Q_fix)*t_ramp_pump_down/2
            q_turbine_up = (np.abs(Wt_intraday)*Q_mult+Q_fix)*t_ramp_turbine_up/2
            q_turbine_down = (np.abs(Wt_intraday)/Q_mult - Q_fix)*t_ramp_turbine_down/2

            # Compute R and x0 from action: 
            # For simplicity, let's say x_pump and x_turbine derived from xhq_opt:
            # In your code you solve a MILP to get x_pump,x_turbine; here we must approximate or assume a known mapping.
            # If your policy directly outputs final pump/turbine schedules (x_pump,x_turbine), then parse them here.
            # For demonstration, let's assume xhq_opt splits into x_pump and x_turbine.
            # This depends heavily on your original action definition.
            # We'll assume xhq_opt: first half = pump power, second half = turbine power (just a guess)
            half = len(xhq_opt)//2
            x_pump = np.array(xhq_opt[:half])
            x_turbine = np.array(xhq_opt[half:])

            R = R + (Delta_ti*beta_pump*np.sum(x_pump) - Delta_ti/beta_turbine*np.sum(x_turbine))
            x0 = x_pump[-1]-x_turbine[-1]

            # Update P_day, P_intraday for next stage:
            P_day = np.concatenate([Wt_day, P_day[:-24].copy()])
            P_intraday = np.concatenate([Wt_intraday, P_intraday[:-96].copy()])
            P_day_sim = np.concatenate([Wt_day, P_day_sim[:-24].copy()])
            P_intraday_sim = np.concatenate([Wt_intraday, P_intraday_sim[:-96].copy()])

            # Compute reward
            reward = (-Delta_td*np.dot(Wt_day,xday_opt)
                      - np.sum(x_pump)*c_grid_fee
                      - Delta_ti*np.dot(Wt_intraday,xhq_opt)
                      + np.dot(q_pump_up, x_pump) - np.dot(q_pump_down, x_pump)
                      - np.dot(q_turbine_up, x_turbine)+ np.dot(q_turbine_down, x_turbine)
                      - 0*Q_start_pump # If start cost not known from action
                      - 0*Q_start_turbine)
            C += reward

        V[m]=C

    EV = np.mean(V)
    print("Average value under learned policy:", EV)
    return EV

if __name__=="__main__":
    evaluate_learned_policy()
