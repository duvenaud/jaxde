from jax import vjp
from jax.flatten_util import ravel_pytree
import jax.numpy as np
from jax import custom_transforms, ad

from jaxde.odeint import odeint

def jvp_odeint(tangent_all, func, y0, t0, t1, func_args):  # future version will take args, and [t]
    flat_args, unravel_args = ravel_pytree(func_args)
    _, unpack = ravel_pytree((y0, y0, flat_args))  # get an un-concatenate function


    def flat_func(y, t, flat_args):
        return func(y, t, *unravel_args(flat_args))

    def augmented_dynamics(augmented_state, t, flat_args):
        y, adj_y, adj_args = unpack(augmented_state)
        dy_dt, vjp_all = vjp(flat_func, y, t, flat_args)
        vjp_a, vjp_t, vjp_args = vjp_all(adj_y)
        return np.concatenate([dy_dt, vjp_a, vjp_args])

    yt, jvp_y, jvp_args = ode_w_linear_part(augmented_dynamics, y0, tangent_all, t0, t1, (flat_args,))
    return yt, jvp_y
ad.primitive_jvps[odeint] = jvp_odeint

#@custom_transforms
def ode_w_linear_part(func, y0, a0, t0, t1, func_args):
    # Just a wrapper around odeint for dynamics that are linear in a0, but not in y0.
    aug_y0, unpack = ravel_pytree((y0, a0))
    aug_ans = odeint(func, aug_y0, np.array([t0, t1]), func_args)
    yt, jvp_y = unpack(aug_ans[1])
    return yt, jvp_y

def odeint_w_linear_part_transpose(cotangent_y0, yt, func, y0, a0, t0, t1):
    assert a0 is None and y0 is not None  # linear in a0 only.
    return ode_w_linear_part(func, yt, cotangent_y0, t1, t0)  # Run ODE backward.
#ad.primitive_transposes[ode_w_linear_part] = odeint_w_linear_part_transpose
