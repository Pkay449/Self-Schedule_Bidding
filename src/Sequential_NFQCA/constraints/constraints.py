# src/Sequential_NFQCA/constraints/constraints.py

import jax.numpy as jnp
from jax import vmap


def build_constraints_single(
    R_val,
    x0,
    Delta_ti,
    beta_pump,
    beta_turbine,
    c_pump_up,
    c_pump_down,
    c_turbine_up,
    c_turbine_down,
    x_min_pump,
    x_max_pump,
    x_min_turbine,
    x_max_turbine,
    Rmax,
):
    """
    Build constraints for a single-step optimization problem involving reservoir state.

    This function uses JAX arrays (jnp) for demonstration, but the structure is similar
    to NumPy-based approaches.

    Parameters
    ----------
    R_val : float
        Some reference or initial reservoir value.
    x0 : float
        Initial water volume or state variable.
    Delta_ti : float
        Time increment or resolution.
    beta_pump : float
        Pump efficiency factor.
    beta_turbine : float
        Turbine efficiency factor.
    c_pump_up : float
        Pump cost factor (up).
    c_pump_down : float
        Pump cost factor (down).
    c_turbine_up : float
        Turbine cost factor (up).
    c_turbine_down : float
        Turbine cost factor (down).
    x_min_pump : float
        Minimum pumping rate.
    x_max_pump : float
        Maximum pumping rate.
    x_min_turbine : float
        Minimum turbine outflow rate.
    x_max_turbine : float
        Maximum turbine outflow rate.
    Rmax : float
        Maximum reservoir capacity.

    Returns
    -------
    A : jnp.ndarray
        Combined inequality constraint matrix for a single step.
    b : jnp.ndarray
        Combined inequality constraint vector for a single step.
    Aeq : jnp.ndarray
        Combined equality constraint matrix for a single step.
    beq : jnp.ndarray
        Combined equality constraint vector for a single step.
    lb : jnp.ndarray
        Variable lower bounds.
    ub : jnp.ndarray
        Variable upper bounds.
    """
    # A1
    A1 = jnp.hstack(
        [
            -jnp.eye(96) + jnp.diag(jnp.ones(95), -1),
            jnp.zeros((96, 96)),
            Delta_ti * beta_pump * jnp.eye(96),
            -Delta_ti / beta_turbine * jnp.eye(96),
            -beta_pump * c_pump_up * jnp.eye(96),
            beta_pump * c_pump_down * jnp.eye(96),
            c_turbine_up / beta_turbine * jnp.eye(96),
            -c_turbine_down / beta_turbine * jnp.eye(96),
            jnp.zeros((96, 96 * 4)),
        ]
    )
    b1 = jnp.zeros(96).at[0].set(-R_val)

    # A2
    Axh = jnp.zeros((96, 24))
    for h in range(24):
        Axh = Axh.at[4 * h : 4 * (h + 1), h].set(-1)

    A2 = jnp.hstack(
        [
            jnp.zeros((96, 96)),
            -jnp.eye(96),
            jnp.eye(96),
            -jnp.eye(96),
            jnp.zeros((96, 96 * 8)),
        ]
    )
    b2 = jnp.zeros(96)

    # A3
    A3 = jnp.hstack(
        [
            jnp.zeros((96, 96 * 2)),
            jnp.eye(96) - jnp.diag(jnp.ones(95), -1),
            jnp.zeros((96, 96)),
            -jnp.eye(96),
            jnp.eye(96),
            jnp.zeros((96, 96 * 6)),
        ]
    )
    b3 = jnp.zeros(96).at[0].set(jnp.maximum(x0, 0))

    # A4
    A4 = jnp.hstack(
        [
            jnp.zeros((96, 96 * 3)),
            jnp.eye(96) - jnp.diag(jnp.ones(95), -1),
            jnp.zeros((96, 96 * 2)),
            -jnp.eye(96),
            jnp.eye(96),
            jnp.zeros((96, 96 * 4)),
        ]
    )
    b4 = jnp.zeros(96).at[0].set(jnp.maximum(-x0, 0))

    Aeq = jnp.vstack([A1, A2, A3, A4])
    beq = jnp.hstack([b1, b2, b3, b4])

    # Constraints for pump and turbine power limits
    A1_pump_turbine = jnp.vstack(
        [
            jnp.hstack(
                [
                    jnp.zeros((96, 96 * 2)),
                    -jnp.eye(96),
                    jnp.zeros((96, 96 * 5)),
                    x_min_pump * jnp.eye(96),
                    jnp.zeros((96, 96 * 3)),
                ]
            ),
            jnp.hstack(
                [
                    jnp.zeros((96, 96 * 2)),
                    jnp.eye(96),
                    jnp.zeros((96, 96 * 5)),
                    -x_max_pump * jnp.eye(96),
                    jnp.zeros((96, 96 * 3)),
                ]
            ),
            jnp.hstack(
                [
                    jnp.zeros((96, 96 * 3)),
                    -jnp.eye(96),
                    jnp.zeros((96, 96 * 5)),
                    x_min_turbine * jnp.eye(96),
                    jnp.zeros((96, 96 * 2)),
                ]
            ),
            jnp.hstack(
                [
                    jnp.zeros((96, 96 * 3)),
                    jnp.eye(96),
                    jnp.zeros((96, 96 * 5)),
                    -x_max_turbine * jnp.eye(96),
                    jnp.zeros((96, 96 * 2)),
                ]
            ),
        ]
    )
    b1_pump_turbine = jnp.zeros(96 * 4)

    # Additional constraints if needed:
    A2_additional = jnp.hstack(
        [
            jnp.zeros((96, 96 * 8)),
            jnp.eye(96) - jnp.diag(jnp.ones(95), -1),
            jnp.zeros((96, 96)),
            -jnp.eye(96),
            jnp.zeros((96, 96)),
        ]
    )
    b2_additional = jnp.zeros(96).at[0].set((x0 > 0).astype(jnp.float32))

    A3_additional = jnp.hstack(
        [
            jnp.zeros((96, 96 * 9)),
            jnp.eye(96) - jnp.diag(jnp.ones(95), -1),
            jnp.zeros((96, 96)),
            -jnp.eye(96),
        ]
    )
    b3_additional = jnp.zeros(96).at[0].set((x0 < 0).astype(jnp.float32))

    A4_additional = jnp.hstack(
        [
            jnp.zeros((96, 96 * 8)),
            jnp.eye(96),
            jnp.eye(96),
            jnp.zeros((96, 2 * 96)),
        ]
    )
    b4_additional = jnp.ones(96)

    A = jnp.vstack([A1_pump_turbine, A2_additional, A3_additional, A4_additional])
    b = jnp.concatenate([b1_pump_turbine, b2_additional, b3_additional, b4_additional])

    # lb and ub
    lb = jnp.concatenate(
        [
            jnp.zeros(96),
            -jnp.inf * jnp.ones(96),
            jnp.zeros(96 * 10),
        ]
    )

    ub = jnp.concatenate(
        [
            Rmax * jnp.ones(96),
            jnp.inf * jnp.ones(96 * 7),
            jnp.ones(96 * 4),
        ]
    )

    return (
        A.astype(jnp.float32),
        b.astype(jnp.float32),
        Aeq.astype(jnp.float32),
        beq.astype(jnp.float32),
        lb.astype(jnp.float32),
        ub.astype(jnp.float32),
    )


def build_constraints_batch(
    states,
    Delta_ti,
    beta_pump,
    beta_turbine,
    c_pump_up,
    c_pump_down,
    c_turbine_up,
    c_turbine_down,
    x_min_pump,
    x_max_pump,
    x_min_turbine,
    x_max_turbine,
    Rmax,
):
    """
    Vectorized building of constraints for multiple states in a batch.

    Parameters
    ----------
    states : ndarray
        A 2D array where each row represents a state [R_val, x0].
    Delta_ti : float
        Time increment or resolution.
    beta_pump : float
        Pump efficiency factor.
    beta_turbine : float
        Turbine efficiency factor.
    c_pump_up : float
        Pump cost factor (up).
    c_pump_down : float
        Pump cost factor (down).
    c_turbine_up : float
        Turbine cost factor (up).
    c_turbine_down : float
        Turbine cost factor (down).
    x_min_pump : float
        Minimum pumping rate.
    x_max_pump : float
        Maximum pumping rate.
    x_min_turbine : float
        Minimum turbine outflow rate.
    x_max_turbine : float
        Maximum turbine outflow rate.
    Rmax : float
        Maximum reservoir capacity.

    Returns
    -------
    tuple
        A, b, Aeq, beq, lb, ub for each state in states, each of shape (batch_size, ...).
    """
    R_val = states[:, 0]  # shape: (batch_size,)
    x0 = states[:, 1]  # shape: (batch_size,)

    # Vectorize the single constraint builder
    A, b, Aeq, beq, lb, ub = vmap(
        build_constraints_single,
        in_axes=(
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )(
        R_val,
        x0,
        Delta_ti,
        beta_pump,
        beta_turbine,
        c_pump_up,
        c_pump_down,
        c_turbine_up,
        c_turbine_down,
        x_min_pump,
        x_max_pump,
        x_min_turbine,
        x_max_turbine,
        Rmax,
    )

    return A, b, Aeq, beq, lb, ub  # Each has shape (batch_size, ...)
