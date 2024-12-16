Below is a cleaned-up pseudocode representation of the given algorithm. It focuses on the core steps and logic, omitting low-level code details for clarity. This format is designed for a poster presentation, providing a high-level, visually appealing overview of the training procedure and how it integrates day-ahead and intraday decisions.

---

**Pseudocode: Hierarchical Actor-Critic for Day-Ahead & Intraday Decision Making**

**Inputs:**
- Offline datasets: 
  - **DA data:** \((S_t^{DA}, A_t^{DA}, R_t^{DA}, S_t^{ID})\)  
  - **ID data:** \((S_t^{ID}, A_t^{ID}, R_t^{ID}, S_{t+1}^{DA})\)
- Neural network models:
  - Q-Networks: \(Q_{DA}(s,a;\theta_{DA})\), \(Q_{ID}(s,a;\theta_{ID})\)
  - Policies: \(\pi_{DA}(s;\phi_{DA})\), \(\pi_{ID}(s;\phi_{ID})\)
- Hyperparameters: learning rate \(\alpha\), discount factor \(\gamma\), batch size \(B\), epochs \(N\)

**Initialize:**
1. **Load DA and ID datasets** from offline files.
2. **Initialize parameters** \(\theta_{DA}, \theta_{ID}, \phi_{DA}, \phi_{ID}\) of Q and policy networks.
3. **Initialize target Q-params:** \(\theta_{DA}^{target} \leftarrow \theta_{DA}\), \(\theta_{ID}^{target} \leftarrow \theta_{ID}\).

**For each epoch \( e = 1,\dots,N \):**

  **Intraday Q-Update:**
  1. For each batch from ID data:
     - Sample \((s_{id}, a_{id}, r_{id}, s_{da}^{next})\).
     - Compute next-day-ahead action: \(a_{da}^{*} = \pi_{DA}(s_{da}^{next}; \phi_{DA})\).
     - Compute ID target:  
       \[
       y^{ID} = r_{id} + \gamma Q_{DA}(s_{da}^{next}, a_{da}^{*}; \theta_{DA}^{target})
       \]
     - Update \(\theta_{ID}\) by minimizing \(\text{MSE}(Q_{ID}(s_{id}, a_{id}; \theta_{ID}), y^{ID})\).

  **Intraday Policy Update:**
  2. Still using the ID batch:
     - Update \(\phi_{ID}\) by maximizing \(Q_{ID}(s_{id}, \pi_{ID}(s_{id}; \phi_{ID}); \theta_{ID})\).
     - Equivalently, minimize \(-Q_{ID}(s_{id}, a_{id}^{\pi}; \theta_{ID})\) where \(a_{id}^{\pi}=\pi_{ID}(s_{id};\phi_{ID})\).

  **Day-Ahead Q-Update:**
  3. For each batch from DA data:
     - Sample \((s_{da}, a_{da}, r_{da}, s_{id}^{next})\).
     - Compute next-intraday action: \(a_{id}^{*} = \pi_{ID}(s_{id}^{next}; \phi_{ID})\).
     - Compute DA target:
       \[
       y^{DA} = r_{da} + \gamma Q_{ID}(s_{id}^{next}, a_{id}^{*}; \theta_{ID}^{target})
       \]
     - Update \(\theta_{DA}\) by minimizing \(\text{MSE}(Q_{DA}(s_{da}, a_{da}; \theta_{DA}), y^{DA})\).

  **Day-Ahead Policy Update:**
  4. Using the DA batch:
     - Update \(\phi_{DA}\) by maximizing \(Q_{DA}(s_{da}, \pi_{DA}(s_{da};\phi_{DA}); \theta_{DA})\).

  **Soft Update of Target Networks:**
  5. Perform soft updates:
     \[
     \theta_{DA}^{target} \leftarrow \tau \theta_{DA} + (1-\tau)\theta_{DA}^{target}
     \]
     \[
     \theta_{ID}^{target} \leftarrow \tau \theta_{ID} + (1-\tau)\theta_{ID}^{target}
     \]

  **Print Progress:**
  6. Print "Epoch e finished."

**End For**

**Action Selection (after training):**
- Given \(s_{da}\): \(a_{da} = \pi_{DA}(s_{da}; \phi_{DA})\) (continuous vector for DA action)
- Given \(s_{id}\): \(a_{id} = \pi_{ID}(s_{id}; \phi_{ID})\) (continuous vector for ID action)

---

**Key Highlights:**
- Uses offline data to learn Q-value approximations and policies.
- Separates day-ahead and intraday value functions, linking them through their respective future stages.
- Continuous actions allow for more realistic and fine-grained control.
- Policies are improved by directly maximizing the expected Q-values, implementing a deterministic policy gradient approach.

This pseudocode elegantly summarizes the training pipeline and the hierarchical actor-critic strategy employed to solve the dynamic, two-stage decision problem of energy storage scheduling and bidding in sequential electricity markets.