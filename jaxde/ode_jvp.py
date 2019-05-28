from jax import vjp, jvp
from jax.flatten_util import ravel_pytree
import jax.numpy as np
from jax import custom_transforms, ad

from jaxde.odeint import odeint


def jvp_odeint(func, (y0, t0, t1, fargs), (tan_y0, tan_t0, tan_t1, tan_fargs)):

    # get an un-concatenate function
    init_state, unpack = ravel_pytree((y0, tan_y0))

    # dummy function to splat fargs
    # TODO: could just expect `func` to take params as tuple
    func_flat = lambda y, t, fargs: func(y, t, *fargs)

    def augmented_dynamics(augmented_state, t):

        # state and senstivity state
        y, a = unpack(augmented_state)

        # combined dynamics
        dy_dt, da_dt = jvp(func_flat, (y, t, fargs), (a, tan_t0, tan_fargs))

        # pack back to give dynamics of augmented_state
        return np.concatenate([dy_dt, da_dt])

    # Solve augmented dynamics
    aug_sol = odeint(augmented_dynamics, init_state, np.array([t0, t1]))
    y0, a0 = unpack(aug_sol[0])
    yt, at = unpack(aug_sol[1])

    # Sensitivities of y(t1) wrt t0 and t1
    jvp_t_total = (tan_t1 - tan_t0) * func(yt, t1, *fargs)

    # Combine sensitivities
    return (np.array([y0, yt]), np.array([tan_y0, at + jvp_t_total]))


#@custom_transforms #TODO: remove this?
def ode_w_linear_part(func, y0, a0, t0, t1, func_args):
    """ Wrapper around odeint for dynamics that are linear in a0 but not y0"""
    aug_y0, unpack = ravel_pytree((y0, a0))
    aug_ans = odeint(func, aug_y0, np.array([t0, t1]), func_args)
    yt, jvp_all = unpack(aug_ans[1])
    return yt, jvp_all


def odeint_w_linear_part_transpose(cotangent_y0, yt, func, y0, a0, t0, t1):
    assert a0 is None and y0 is not None  # linear in a0 only.
    return ode_w_linear_part(func, yt, cotangent_y0, t1,
                             t0)  # Run ODE backward.

    #ad.primitive_transposes[ode_w_linear_part] = odeint_w_linear_part_transpose #TODO: remove this too?
