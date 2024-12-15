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

# =====================
# Helper Functions
# =====================

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

def compute_weights(eng, phi, Y, weights_vector):
    weights = VRx_weights(phi, Y, weights_vector)
    return weights

def build_and_solve_intlinprog(eng, f, A, b, Aeq, beq, lb, ub, intcon, options):
    matlab_f = matlab.double((-f).tolist())  # MATLAB: minimize
    matlab_A = matlab.double(A.tolist())
    matlab_b = matlab.double(b.tolist())
    matlab_Aeq = matlab.double(Aeq.tolist())
    matlab_beq = matlab.double(beq.tolist())
    matlab_lb = matlab.double(lb.tolist())
    matlab_ub = matlab.double(ub.tolist())
    matlab_intcon = matlab.double((intcon+1).tolist())
    xres, fvalres = eng.intlinprog(matlab_f, matlab_intcon, matlab_A, matlab_b,
                                   matlab_Aeq, matlab_beq, matlab_lb, matlab_ub,
                                   options, nargout=2)
    x_opt = np.array(xres).flatten()
    fval = float(fvalres)
    return x_opt, fval

