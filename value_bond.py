import jax
import jax.numpy as jnp

def schedule_from_end(Te, freq=2, Ts=0.0, eps=1e-12, dtype=jnp.float32):
    """
    Generate coupon schedule backwards from Te down to Ts.
    Returns strictly increasing coupon times ending at Te, with accruals.
    """
    f = jnp.array(freq, dtype=dtype)

    # full coupon spacing
    step = 1.0 / f
    n_full = jnp.floor((Te - Ts) * f + eps).astype(int)

    # backward full coupons
    times_rev = Te - step * jnp.arange(n_full, dtype=dtype)
    times = jnp.flip(times_rev)

    # include Ts if it's not aligned
    if Ts < times[0] - eps:
        times = jnp.concatenate([jnp.array([Ts], dtype=dtype), times])

    # accruals are differences between times
    accruals = jnp.diff(times, prepend=Ts)

    return times, accruals

def par_yield(Ts, Te, A, d, Sigma, X0, p, freq=2):
    """
    Par yield at t=0 for a bond paying 'freq' coupons/year from Ts to Te.
    Schedule is generated backwards from Te.
    """
    dtype = A.dtype
    Ts = jnp.asarray(Ts, dtype=dtype)
    Te = jnp.asarray(Te, dtype=dtype)

    pay_times, accruals = schedule_from_end(Te, freq=freq, Ts=Ts, dtype=dtype)

    # Discount factors at payment dates
    P_vec, _ = v_discount_and_yield_linear(pay_times, A, d, Sigma, X0, p)

    # annuity includes all coupons (including final if aligned)
    annuity = jnp.sum(P_vec * accruals)
    P_Te    = P_vec[-1]  # principal

    return (1.0 - P_Te) / (annuity + 1e-16)

def value_bond(Ts, Te, coupon, A, d, Sigma, X0, p, freq=2):
    """
    Bond PV at t=0 for coupon-paying bond.
    Uses actual accruals from the schedule (handles stubs correctly).
    """
    dtype = A.dtype
    Ts = jnp.asarray(Ts, dtype=dtype)
    Te = jnp.asarray(Te, dtype=dtype)

    pay_times, accruals = schedule_from_end(Te, freq=freq, Ts=Ts, dtype=dtype)

    # Discount factors at payment dates
    P_vec, _ = v_discount_and_yield_linear(pay_times, A, d, Sigma, X0, p)

    # Coupon leg: coupon * accrual * discount
    coupons = jnp.sum(P_vec * coupon * accruals)

    # Principal
    principal = P_vec[-1]

    return coupons + principal
