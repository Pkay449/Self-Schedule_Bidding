# src/Sequential_NFQCA/training/optimizers.py

import optax


def get_optimizer(learning_rate: float):
    return optax.adam(learning_rate)
