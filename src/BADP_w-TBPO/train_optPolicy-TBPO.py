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
    
data
# %%
plt.plot(data['reward'])

# %%
