# %% [markdown]
# # Energy storage bidding strategies using approximate dynamic programming
#
# ### Paul Kelendji, Uday Kapur, Carlone Scott

# %% [markdown]
# ## Introduction

# %% [markdown]
# Energy storage systems help stabilize electricity grids by leveraging price fluctuations in day-ahead and intraday markets. Optimizing bidding decisions in these markets is complex due to high-dimensional price forecasts and operational constraints. Our approach builds on the approximate dynamic programming (ADP) approach by Finnah et al., aiming to improve computational efficiency and scalability for handling high-dimensional state spaces.

# %% [markdown]
# ### Imports

# %%


# %% [markdown]
# ### Simulation Parameters


# %%



# %% [markdown]
# ### Helper Functions
#
# - `badp_weights`: Computes weights for day-ahead and intraday price influence using forecast and historical coefficients.
# - `VRx_weights`: Optimizes weights via quadratic programming and kernel-based distance adjustments.
# - `sample_price_day`: Calculates expected values and covariance matrix for day-ahead prices based on observed data and time step.
# - `sample_price_intraday`: Calculates expected values and covariance matrix for intraday prices using day-ahead and intraday data.
# - `generate_scenarios`: Simulates daily and intraday price scenarios for multiple forecasts with randomization.
# - `compute_weights`: Wrapper for VRx-based weight computation.
# - `build_and_solve_intlinprog`: Sets up and solves integer linear programming problems via MATLAB integration.
# - `linear_constraints_train`: Builds linear constraints for optimization problems involving pumps and turbines.
# - `build_constraints_single`: Constructs constraints for a single optimization instance using JAX.
# - `build_constraints_batch`: Vectorizes single constraint construction for batch processing.
# - `reverse_price_series`: Reverses a time series of prices day-by-day for analysis or simulation.


# %% [markdown]
# # BADP Approach (Baseline)
#
# The approached presented by Finnah et. al, presents a backwards approximate dynamic programming (BADP) approach for self-scheduling bidding in German day-ahead and intraday electricity auction markets. The method focuses on profit-maximizing energy storage systems, particularly pumped hydro storage (PHS), to exploit price differences between these markets. The authors modeled the problem as a dynamic programming (DP) problem considering the characteristics of the day-ahead (hourly) and intraday (15-minute) markets.They incorporated high-dimensional electricity price forecasts as a state variable, and accounting for price volatility and market dynamics. The steps involved are described below:
#
# **Backwards Approximate Dynamic Programming (BADP):**
#
# - Utilized a discretized state space for computational feasibility.
# - Derived value function approximations iteratively using sampled price paths to estimate future states without transition probabilities.
# - Weighted regression techniques incorporated price importance in forecasts, refining accuracy.
#
# **Policy Design:**
#
# - Simultaneously optimized decisions for both markets, with intraday adjustments based on real-time day-ahead results.
#
# **Numerical Study:**
#
# - Benchmarked BADP against other state-of-the-art methods, demonstrating superior handling of high-dimensional price processes.
# - Integrated trading in both markets outperformed sequential trading strategies.
#
# **Scalability:**
#
# - The approach effectively managed the curse of dimensionality by simplifying price processes and leveraging approximate algorithms.
#
# The approach the authors present emphasizes practical implementation by handling real-world data, balancing computational feasibility, and optimizing storage decisions in volatile market environments.
#
# This section below trains the BADP (Backward Approximate Dynamic Programming) model by simulating price scenarios, computing expected values and constraints, and solving optimization problems iteratively across scenarios and time steps. The backward pass computes value functions (Vt) using dynamic programming and integrates intraday and day-ahead price influences. Results are saved for further analysis. This serves as the baseline with which we compare our approach in the follwoing sections.

# %% [markdown]
# ## Generate offline samples
#
# This section evaluates the learned value function and generates offline datasets for day-ahead and intraday price optimization. It simulates scenarios, computes rewards, and saves state-action transitions in datasets to support offline learning and analysis. The evaluation calculates the expected value (EV) for the policy's performance.

# %% [markdown]
# ## Evaluate on held out test set
#
# This section evaluates the trained policy on a held-out test set using historical price data. The process involves solving optimization problems for day-ahead and intraday decisions and computing performance metrics. A typical day in the electricity trading market is simulated by first forecasting day-ahead prices, generating optimal actions, and then using the equilibrium price of the day-ahead market to determine intraday market decisions. The evaluation calculates the expected value (EV) and backtest statistics, including mean, standard deviation, and total reward. Paths and storage trackers are saved for further analysis and visualization.

# %% [markdown]
# ## Results Visualization - BADP Approach
#
# The plots below demonstrate the interplay between energy storage, market prices, and operational decisions, highlighting the effectiveness of the policy in managing a complex energy system. The system responds dynamically to market signals and operational constraints, ensuring balanced performance.

# %% [markdown]
# # Our Approach - NFQCA
#
# The NFQCA (Neural Fitted Q-Continuous Action) approach builds upon the BAPD baseline by introducing improved action-space modeling and constraint handling through Quadratic Programming (QP) projections and policy regularization. Below is a brief description of key elements of our approach:
#
# #### 1. **QP Projection (`qp_projection`)**
# - Ensures that raw actions generated by the policy network respect predefined bounds and constraints.
# - Solves a QP to minimize the distance between raw actions and feasible actions while enforcing equality, inequality, and bound constraints.
# - Includes relaxation parameters for numerical stability when enforcing constraints.
#
# #### 2. **Offline Data Loading**
# - The `load_offline_data` function loads the offline data (day-ahead and intraday) for supervised learning and reinforcement learning.
# - These datasets consist of state-action-reward-next-state tuples essential for training and evaluating Q-functions and policies.
#
# #### 3. **Neural Network Models**
# - **QNetwork**: Predicts Q-values for state-action pairs using a feedforward network, enabling the learning of value functions for day-ahead (`Q_DA`) and intraday (`Q_ID`) markets.
# - **PolicyDA**: Represents the day-ahead policy, constrained to lie within specific bounds using a sigmoid transformation.
# - **PolicyID**: Defines the intraday policy with complex constraints enforced by combining QP-based projections and penalty-based learning mechanisms.
#
# #### 4. **Constraint Penalty Enforcement**
# - The `update_policy_id_with_penalty` function penalizes actions that violate operational constraints during policy optimization.
# - Leverages a batch-constraint generation function (`build_constraints_batch`) for efficient computation.
#
# #### 5. **Dual Policy Optimization**
# - **Day-Ahead Optimization**: Focuses on optimizing actions for strategic, long-term decisions based on day-ahead price forecasts.
# - **Intraday Optimization**: Refines decisions based on real-time intraday price updates, incorporating tighter operational constraints.
#
# #### 6. **Training Process**
# - Updates Q-functions (`Q_DA` and `Q_ID`) using Bellman targets derived from rewards and next-step Q-values.
# - Trains policies (`PolicyDA` and `PolicyID`) to maximize their respective Q-functions while adhering to operational constraints.
# - Uses a soft-update mechanism to stabilize the learning of target networks for both Q-functions.
#
# #### Improvements were made using our NFQCA approach by incorporating:
# - Robust constraint handling.
# - Enhanced action representation through QP projections.
# - Dual policy optimization tailored for hierarchical decision-making.
#
# ### Architecture
#
# ![image-2.png](attachment:image-2.png)

# %% [markdown]
# ### Define Q and Policy networks

# %% [markdown]
# ### Helper functions for NFQCA


# %% [markdown]
# ### Load data

# %% [markdown]
# ### Bounds for Intraday

# %% [markdown]
# ### Initialize Q and Policy networks

# %% [markdown]
# ## Training loop for Multi-NFQCA

# %% [markdown]
# ## Evaluating Learned Policies with NFQCA
#
# We evaluate the performance of learned policies (day-ahead and intra-day) under the NFQCA framework using a held-out test dataset. It simulates a day-ahead and intraday market environment while considering operational constraints and energy market dynamics.


# %% [markdown]
# ## Results Visualization - NFQCA
#
# The plots below illustrate the performance of the NFQCA approach in managing energy storage and operational constraints. Key elements such as market prices, intraday actions, and pump/turbine operations are visualized, showcasing the model's ability to adapt dynamically to market signals and system constraints. This visualization highlights how the NFQCA policy efficiently balances energy resources while adhering to operational boundaries, improving upon baseline of the BAPD approach.

# %% [markdown]
# # Alternative Approaches Explored

# %% [markdown]
# In this work, we primarily relied on **penalty-based constraint enforcement** to ensure the actions generated by the policy remained within permissible operational bounds. However, we also investigated other strategies to handle the complex constraints and potentially improve training stability and computational efficiency.
#
# 1. **Projection Methods:**
#    - **Quadratic Programming (QP) Projection:** Given the Intraday policy’s intricate constraints, we explored using a QP-based projection step. After the policy network produced raw actions, these actions would be **projected** onto the feasible set defined by the constraints, guaranteeing operational compliance.
#    - **Advantages:** Projection methods **guarantee** feasibility, preventing actions that violate constraints. This is particularly attractive in real-world scenarios where safety or reliability constraints must be strictly upheld.
#    - **Limitations:** Unfortunately, our experimental results revealed that the QP-based projection incurred **significant computational overhead** and led to **numerical instability**, thus hindering the scalability and convergence of the approach under our problem settings.
#
# 2. **Residual Learning:**
#    - **Motivation:** We noted that our environment’s state dimension **alternates every two steps**. Hence, we initially implemented two NFQCAs (Neural Fitted Q with Continuous Actions): one for the Day-Ahead states and one for the Intraday states. This duplication arose from the distinct input structures required at each stage.
#    - **Potential for a Single NFQCA:** With **residual learning**, we could have designed a single NFQCA architecture that learns Q-values for both Day-Ahead and Intraday states in a unified manner. This approach might leverage **shared representations** across states, reducing the need for maintaining two separate networks.
#    - **Benefits and Challenges:** Residual learning could simplify the overall architecture and potentially **improve generalization** by sharing parameters. However, it also introduces complexity in determining how the residual blocks interact across different state representations. Additional experimentation would be needed to confirm whether this consolidated approach outperforms separate models in practice.
#
# Overall, while **penalty-based methods** and **dual policy optimization** formed the core of our solution, **projection methods** and **residual learning** highlight possible avenues for future exploration. Further research may focus on overcoming the computational and numerical issues identified in the projection approach, as well as investigating whether a single, residual-based NFQCA framework can effectively handle both Day-Ahead and Intraday state dimensions without compromising performance or stability.

# %% [markdown]
#

# %% [markdown]
#
