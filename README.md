# Self Schedule Bidding

**Context & Motivation:**  
Energy markets commonly require participants to commit to energy delivery before the actual delivery time. This practice leads to separate day-ahead and intraday markets, each with their own timescales, price structures, and uncertainties. High-frequency intraday markets introduce volatility that can be leveraged by flexible assets like pumped-storage facilities. Previous work researched scenario where dynamic programming and approximate methods were used to address the complexity of bidding and operations in these markets. Our project aims to build upon and improve the methods and models described in that work, exploring enhanced forecasting techniques, more flexible optimization frameworks, and scalability to higher dimensions.

## Our Vision and Contributions

While the original approach provided a robust framework for dealing with complexity and uncertainty in energy markets, we see opportunities for improvement:

- **Richer Forecasting Models:**  
  We plan to integrate advanced statistical and machine learning techniques to generate more accurate and granular price distributions for both day-ahead and intraday markets.

- **Enhanced Optimization Algorithms:**  
  Beyond classical approximate dynamic programming, we will explore reinforcement learning, scenario reduction techniques, and hybrid approaches to manage the high dimensionality and complexity of the value function.

- **Scalability & Computational Efficiency:**  
  We aim to reduce computational overhead through more efficient solvers, parallelization strategies, and improved state-space discretizations.

- **Robustness & Stress Testing:**  
  Incorporating stress testing and robust optimization methods to ensure the model’s recommendations remain effective under extreme price scenarios.

---

## Repository Structure

TBA

## Key Components:

**Forecasting Modules:**

`sample_price_day.py` and `sample_price_intraday.py` to generate price forecasts.

**### **Optimization & Approximation:**

`badp_w.py`, `badp_weights.py`, and `VRx_weights.py` implement the dynamic programming and approximation methods.

**Data & Config:**

The `Data` folder and `config.json` store historical price data and configuration parameters.

Getting Started
Dependencies:

```bash
pip install -r requirements.txt
```
TBC







