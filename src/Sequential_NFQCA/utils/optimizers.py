# src/Sequential_NFQCA/utils/optimizers.py

import jax

def soft_update(target_params, online_params, tau=0.005):
    return jax.tree_util.tree_map(
        lambda tp, op: tp * (1 - tau) + op * tau, target_params, online_params
    )