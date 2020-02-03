from jax import vjp, jvp
from jax.flatten_util import ravel_pytree
import jax.numpy as np
from jax import ad, ad_util

from jaxde.odeint import odeint


def jvp_odeint(primals, tangents, func=None):
    (y0, ts, fargs) = primals
    (tan_y0, tan_ts, tan_fargs) = tangents
    t0, t1 = ts
    tan_t0, tan_t1 = tan_ts

    print("ts z: ",ts)
    
    # TODO: maybe avoid instantiating zeros in some cases
    tan_y0 = ad.instantiate_zeros(y0, tan_y0)
    tan_t0 = ad.instantiate_zeros(t0, tan_t0)
    if tan_fargs is ad_util.zero:
      zeros = (ad_util.zero,) * len(fargs)
      tan_fargs = tuple(map(ad.instantiate_zeros, fargs, zeros))

    # get an un-concatenate function
    init_state, unpack = ravel_pytree((y0, tan_y0))

    def augmented_dynamics(augmented_state, t, fargs):

        # state and senstivity state
        y, a = unpack(augmented_state)
        a = ad.instantiate_zeros(y, a)

        # combined dynamics
        dy_dt, da_dt = jvp(func, (y, t, fargs), (a, tan_t0, tan_fargs))

        # pack back to give dynamics of augmented_state
        return np.concatenate([dy_dt, da_dt])

    # Solve augmented dynamics
    aug_sol = odeint(init_state, np.array([t0, t1]), fargs, func=augmented_dynamics)
    # aug_sol = odeint(augmented_dynamics,init_state, np.array([t0, t1]), fargs)
    yt, at = unpack(aug_sol[1])

    # Sensitivities of y(t1) wrt t0 and t1
    jvp_t_total = (tan_t1 - tan_t0) * func(yt, t1, fargs)

    # Combine sensitivities
    tan_yt = jvp_t_total if at is ad_util.zero else at + jvp_t_total
    return (np.array([y0, yt]), np.array([tan_y0, tan_yt]))

ad.primitive_jvps[odeint.prim] = jvp_odeint
