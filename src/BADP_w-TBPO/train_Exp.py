# %%[markdown]
# #### Neural Fitted Q-Iteration with Continuous Actions (NFQCA)

# **Input** MDP $(S, A, P, R, \gamma)$, base points $\mathcal{B}$, Q-function $q(s,a; \boldsymbol{\theta})$, policy $d(s; \boldsymbol{w})$

# **Output** Parameters $\boldsymbol{\theta}$ for Q-function, $\boldsymbol{w}$ for policy

# 1. Initialize $\boldsymbol{\theta}_0$, $\boldsymbol{w}_0$
# 2. **for** $n = 0,1,2,...$ **do**
#     1. $\mathcal{D}_q \leftarrow \emptyset$
#     2. For each $(s,a,r,s') \in \mathcal{B}$:
#         1. $a'_{s'} \leftarrow d(s'; \boldsymbol{w}_n)$
#         2. $y_{s,a} \leftarrow r + \gamma q(s', a'_{s'}; \boldsymbol{\theta}_n)$
#         3. $\mathcal{D}_q \leftarrow \mathcal{D}_q \cup \{((s,a), y_{s,a})\}$
#     3. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D}_q)$
#     4. $\boldsymbol{w}_{n+1} \leftarrow \texttt{minimize}_{\boldsymbol{w}} -\frac{1}{|\mathcal{B}|} \sum_{(s,a,r,s') \in \mathcal{B}} q(s, d(s; \boldsymbol{w}); \boldsymbol{\theta}_{n+1})$
# 3. **return** $\boldsymbol{\theta}_n$, $\boldsymbol{w}_n$
# %%
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
import os
# %%

# load src/BADP_w-TBPO/Data/offline_dataset_authors_data.pkl
path = 'Results/offline_dataset.pkl'
with open(path, 'rb') as f:
    data = pkl.load(f)
    

# %%


import jax
import jax.numpy as jnp
import optax
import pickle as pkl
import numpy as np

# Load offline dataset
with open('Results/offline_dataset.pkl', 'rb') as f:
    data = pkl.load(f)

# Extract arrays from DataFrame: assumes data is a DataFrame with correct columns.
states = np.stack(data['state'].values)
actions = np.stack(data['action'].values)
rewards = np.array(data['reward'].values)
next_states = np.stack(data['next_state'].values)

# Convert to jax arrays
states = jnp.array(states)
actions = jnp.array(actions)
rewards = jnp.array(rewards)
next_states = jnp.array(next_states)

# Set hyperparams
gamma = 0.99
learning_rate = 1e-3
num_iterations = 1000
batch_size = 64

num_samples = states.shape[0]
state_dim = states.shape[1]
action_dim = actions.shape[1]

# Define MLP functions for Q and policy
def mlp(params, x):
    # params = [(W1,b1), (W2,b2), ...]
    for W,b in params[:-1]:
        x = jnp.tanh(jnp.dot(x,W)+b)
    W,b = params[-1]
    return jnp.dot(x,W)+b

def init_mlp(key, in_dim, out_dim, hidden_sizes=(64,64)):
    params = []
    k = key
    dims = [in_dim] + list(hidden_sizes) + [out_dim]
    for i in range(len(dims)-1):
        k,subk = jax.random.split(k)
        W = jax.random.normal(subk, (dims[i], dims[i+1]))*0.1
        b = jnp.zeros((dims[i+1],))
        params.append((W,b))
    return params

# Initialize Q-network and Policy-network
key = jax.random.PRNGKey(0)
q_params = init_mlp(key, state_dim+action_dim, 1)
key, subkey = jax.random.split(key)
policy_params = init_mlp(subkey, state_dim, action_dim)

# Define forward functions
def q_function(params, s, a):
    sa = jnp.concatenate([s,a], axis=-1)
    return mlp(params, sa)

def policy(params, s):
    # For simplicity, a deterministic policy with tanh last layer to bound actions if needed
    x = s
    for W,b in params[:-1]:
        x = jnp.tanh(jnp.dot(x,W)+b)
    W,b = params[-1]
    # For continuous action, maybe we want to scale output by a factor
    a = jnp.dot(x,W)+b
    return a

# Optimizers
q_opt = optax.adam(learning_rate)
policy_opt = optax.adam(learning_rate)
q_opt_state = q_opt.init(q_params)
policy_opt_state = policy_opt.init(policy_params)

@jax.jit
def q_loss_fn(q_params, s, a, y):
    q_pred = q_function(q_params, s, a).squeeze(-1)
    return jnp.mean((q_pred - y)**2)

@jax.jit
def policy_loss_fn(policy_params, q_params, s):
    a = policy(policy_params, s)
    q_val = q_function(q_params, s, a)
    return -jnp.mean(q_val)

q_grad_fn = jax.grad(q_loss_fn)
policy_grad_fn = jax.grad(policy_loss_fn)

def sample_batch(batch_size):
    idx = np.random.randint(0, num_samples, size=batch_size)
    return (states[idx], actions[idx], rewards[idx], next_states[idx])

for iter in range(num_iterations):
    # Compute target y_{s,a}
    # First we need a'
    a_prime = policy(policy_params, next_states)
    q_next = q_function(q_params, next_states, a_prime).squeeze(-1)
    y = rewards + gamma*q_next

    # We'll do Q fitting
    # Sample batch
    s_b, a_b, r_b, s_next_b = sample_batch(batch_size)
    a_prime_b = policy(policy_params, s_next_b)
    q_next_b = q_function(q_params, s_next_b, a_prime_b).squeeze(-1)
    y_b = r_b + gamma*q_next_b

    q_g = q_grad_fn(q_params, s_b, a_b, y_b)
    updates, q_opt_state = q_opt.update(q_g, q_opt_state, q_params)
    q_params = optax.apply_updates(q_params, updates)

    # Policy improvement step
    p_g = policy_grad_fn(policy_params, q_params, s_b)
    updates, policy_opt_state = policy_opt.update(p_g, policy_opt_state, policy_params)
    policy_params = optax.apply_updates(policy_params, updates)

    if iter % 100 == 0:
        # Compute a loss for logging
        loss_val = q_loss_fn(q_params, s_b, a_b, y_b)
        pol_loss_val = policy_loss_fn(policy_params, q_params, s_b)
        print(f"Iter {iter}, Q-loss: {loss_val:.4f}, Policy-loss: {pol_loss_val:.4f}")

print("Training complete.")

# save q_params and policy_params
with open('Results/q_params.pkl', 'wb') as f:
    pkl.dump(q_params, f)
    
with open('Results/policy_params.pkl', 'wb') as f:
    pkl.dump(policy_params, f)
    

# q_params and policy_params now represent the learned Q and policy.

# %%
