Below is a thorough and scientific documentation on how the given algorithm works and how it relates to and solves the dynamic programming (DP) problem described in Section 3 of the referenced theoretical framework. We will break down the problem, the algorithmic approach, and how the provided code implements an approximate solution using Reinforcement Learning (RL) and function approximation techniques.

---

## Problem Context and Dynamic Programming Formulation

### The Problem

The storage owner operates a Pumped Hydro Storage (PHS) unit and participates in two sequential electricity markets:

1. **Day-Ahead Market (DA):**  
   Each day \( t \), the storage owner must commit to 24 hourly volumes \( x_t^{DA} = (x_{t,0}^{DA}, \dots, x_{t,23}^{DA}) \) for the next day’s delivery. After these commitments are made, the day-ahead clearing prices \( P_{t+1}^{DA} \) are realized and the scheduled volumes come into effect.

2. **Intraday Market (ID):**  
   After the day-ahead prices are known, and before the actual delivery hour, the storage owner can adjust the schedule in the intraday market with a finer granularity (e.g., quarter-hourly intervals). Thus, the intraday decision \( x_t^{ID} \) refines or corrects the initial commitments made in the day-ahead stage. Intraday market clearing prices \( P_{t+1}^{ID} \) are realized afterward.

The objective is to maximize the expected total profit over a horizon \( T \) days. The profit at each stage depends on the chosen actions (volumes), the market clearing prices, ramping costs, start-up costs, and other grid-related costs as described in Sections 3.3 and 3.4 of the theoretical framework.

### States, Actions, and Dynamics

- **States (\( S_t^{DA}, S_t^{ID} \)):**  
  The state includes the storage level, the starting operation point, and a history of past day-ahead and intraday prices needed for forecasting. After the day-ahead decision and the realization of day-ahead prices, the state transitions to an intraday state \( S_t^{ID} \). After the intraday decision and realization of intraday prices, the system transitions to the next day’s day-ahead state \( S_{t+1}^{DA} \).

- **Actions (\( x_t^{DA} \) and \( x_t^{ID} \)):**  
  At the day-ahead stage, actions are the hourly commitments for the next day. At the intraday stage, actions adjust the power output/consumption at a finer time scale (quarter-hourly).

- **Exogenous Information (\( W_t^{DA}, W_t^{ID} \)):**  
  After the DA decision, \( P_{t+1}^{DA} \) is revealed. After the ID decision, \( P_{t+1}^{ID} \) is revealed. These prices depend on historical prices and day-of-week effects, and are modeled stochastically.

- **Contribution Functions and Transitions:**  
  Each decision stage yields immediate profits or costs. The system transitions according to the described mechanics (storage dynamics, ramping, next-day price forecasts).

### Goal

The dynamic program’s Bellman equations (e.g., Equations (28) and (29)) express the value functions for the day-ahead and intraday states. Ideally, one would solve these Bellman equations exactly. However, the dimensionality of the state and action spaces makes exact solution intractable.

---

## Approximate Dynamic Programming and Reinforcement Learning Approach

Since the exact DP solution is intractable due to high dimensionality and complexity, the code uses Reinforcement Learning (RL) methods to approximate the value functions and derive policies. In particular, it employs:

- **Q-Networks:** Two Q-functions are learned, \( Q_{DA}(S_t^{DA}, X_t^{DA}) \) and \( Q_{ID}(S_t^{ID}, X_t^{ID}) \), to represent the expected future cumulative reward (profit) from each decision point onward.
- **Policy Networks:** Two policies \( \pi_{DA}(S_t^{DA}) \) and \( \pi_{ID}(S_t^{ID}) \) are learned. These map states to actions directly and are parameterized by neural networks. The actions are now continuous, capturing the volume decisions in a continuous domain rather than discrete.

The algorithm thus fits into a hierarchical actor-critic RL approach, where the critics are the Q-networks and the actors are the policy networks.

---

## Key Components of the Code

### 1. Data Loading

The provided code loads offline datasets `offline_dataset_day_ahead.pkl` and `offline_dataset_intraday.pkl`. These contain historical transitions:

- Day-ahead dataset: \((S_t^{DA}, X_t^{DA}, R_t^{DA}, S_t^{ID})\)
- Intraday dataset: \((S_t^{ID}, X_t^{ID}, R_t^{ID}, S_{t+1}^{DA})\)

Here, \( R_t^{DA} \) and \( R_t^{ID} \) are immediate rewards (profits/losses). The datasets are used to perform offline RL learning.

### 2. Neural Network Architecture

- **Q Networks (Critics):**  
  Two Q-networks, `QNetworkDA` and `QNetworkID`, approximate \( Q_{DA} \) and \( Q_{ID} \) respectively. Each takes as input a state and an action and outputs a single scalar Q-value. They are fully connected networks with ReLU activations.

- **Policy Networks (Actors):**  
  `PolicyDA` and `PolicyID` output continuous action vectors given the state. They are also MLPs with ReLU activations. Instead of producing discrete actions, they now produce real-valued vectors representing the chosen market volumes continuously.

### 3. Training Methodology

The training loop iterates over epochs and uses batches from the loaded offline datasets. For each transition, it updates both the critics and the actors:

- **Critic Updates (Q Learning):**
  - **Intraday Critic (`Q_ID`) Update:**  
    The target Q-value for intraday is:
    \[
    Q_{target}^{ID} = R_t^{ID} + \gamma Q_{DA}(S_{t+1}^{DA}, \pi_{DA}(S_{t+1}^{DA}))
    \]
    This means the intraday Q-value depends on the immediate intraday reward plus the future value from the next day-ahead state. The future value is approximated by taking the next day-ahead action from the current day-ahead policy and evaluating it in the target Q-DA network.

  - **Day-Ahead Critic (`Q_DA`) Update:**  
    Similarly:
    \[
    Q_{target}^{DA} = R_t^{DA} + \gamma Q_{ID}(S_t^{ID}, \pi_{ID}(S_t^{ID}))
    \]
    The day-ahead Q-value depends on the immediate day-ahead reward plus the expected future intraday value. The future intraday action is chosen by the current intraday policy.

  MSE loss between the predicted Q-values and these targets is minimized using gradient descent (via JAX and Optax).

- **Actor Updates (Policy Improvement):**
  Once the critics (Q-networks) are trained to approximate the value functions, the policies are improved. The policy updates follow a deterministic policy gradient idea:

  For the day-ahead policy:
  \[
  \nabla_{\theta_{DA}} J \approx \mathbb{E}[\nabla_{\theta_{DA}} Q_{DA}(S_t^{DA}, \pi_{DA}(S_t^{DA}))]
  \]
  The code performs gradient ascent on the Q-value with respect to the policy parameters. In other words, it adjusts the policy parameters to produce actions that yield higher Q-values.

  Similarly for the intraday policy:
  \[
  \nabla_{\theta_{ID}} J \approx \mathbb{E}[\nabla_{\theta_{ID}} Q_{ID}(S_t^{ID}, \pi_{ID}(S_t^{ID}))]
  \]

  By doing so, both policies learn to produce actions that maximize the expected returns as indicated by their respective critics.

### 4. Target Networks and Stability

To stabilize learning, target networks (`q_da_target_params` and `q_id_target_params`) are used. They are slowly updated versions of the main Q-network parameters, reducing oscillations and stabilizing training. The soft update step:
\[
\theta_{target} \leftarrow \tau \theta_{online} + (1-\tau) \theta_{target}
\]

### 5. Continuous Actions

Originally, actions were discrete. By treating them as continuous, the approach aligns more naturally with the actual physical decisions of ramping power up/down continuously. This reduces the complexity of discretization and `argmax` selection. Instead, the policy directly outputs a feasible continuous vector of actions.

---

## How This Algorithm Solves the Dynamic Program

The DP problem described in Section 3 is to find an optimal policy \(\pi\) that maximizes expected cumulative profit. The Bellman equations (28) and (29) define value functions for the day-ahead and intraday stages. In principle, solving these equations exactly is challenging due to the large state and action spaces, stochastic prices, and complicated constraints.

**The code addresses this complexity by:**

1. **Approximate Dynamic Programming via RL:**
   Instead of enumerating states and actions or performing backward recursion, the algorithm uses neural networks to approximate Q-values. These Q-values serve as proxies for the value functions \( V_t^{DA} \) and \( V_t^{ID} \).

2. **Two-Stage Hierarchical Q-Learning:**
   By having separate Q-networks (one for day-ahead and one for intraday) and policies for each stage, the algorithm effectively decomposes the problem into two connected RL subproblems that alternate each day:
   - At the DA stage: The Q_DA network approximates:
     \[
     V_t^{DA}(S_t^{DA}) = \max_{x_t^{DA}} \mathbb{E}[C_t^{DA} + V_t^{ID}(S_t^{ID}) | S_t^{DA}]
     \]
     The policy DA attempts to select \( x_t^{DA} \) that maximizes the approximate Q_DA values.

   - At the ID stage: The Q_ID network approximates:
     \[
     V_t^{ID}(S_t^{ID}) = \max_{x_t^{ID}} \mathbb{E}[C_t^{ID} + V_{t+1}^{DA}(S_{t+1}^{DA}) | S_t^{ID}]
     \]
     The policy ID attempts to pick \( x_t^{ID} \) that maximizes Q_ID.

3. **Stochastic Approximation:**
   The Q-value updates and policy improvements rely on samples from offline datasets. Over many updates, the neural networks learn to generalize and approximate the mapping from states to values and from states to optimal actions.

4. **Future Value Incorporation:**
   Each update includes the discounted future value from the next stage:
   - Intraday Q-target uses next day-ahead Q-value.
   - Day-ahead Q-target uses next intraday Q-value.

   This enforces consistency with the Bellman equations, ensuring the learned Q-values move toward a fixed point representing the optimal value functions.

---

## Interpretation and Practical Use

By training these networks, the algorithm learns a pair of policies \(\pi_{DA}\) and \(\pi_{ID}\) that approximately solve the dynamic program described. The resulting policies can be used to make near-optimal day-ahead and intraday decisions, balancing current costs/revenues with future opportunities. Over time, this should lead to improved operational strategies and higher expected profits.

The method is essentially a practical ADP/RL solution where classical DP methods fail due to dimensionality and complexity. Instead of discretizing actions or solving large-scale linear or nonlinear programs for each state, we rely on neural approximators and iterative updates guided by past experience (offline dataset) to find a suitable policy.

---

## Conclusion

This algorithm embodies a reinforcement learning approach to solving a complex, high-dimensional, stochastic dynamic programming problem faced by a pumped hydro storage operator participating in sequential energy markets. By using continuous policy networks and Q-value function approximation:

- It transforms a two-stage problem (day-ahead followed by intraday decisions) into a manageable RL problem.
- Each critic (Q-network) encodes the expected future returns for given actions.
- Each actor (policy network) learns to select actions that maximize these learned value functions.
- The combination of these steps and iterative updates leads toward an approximate solution of the original DP problem, allowing the storage owner to derive near-optimal bidding strategies that account for future price uncertainties, ramping constraints, and operational costs in both day-ahead and intraday markets.

In summary, the code’s approach is a form of hierarchical actor-critic RL applied to a complicated energy storage optimization problem, providing a scalable and flexible method to approximate the optimal policies defined by the given dynamic program.