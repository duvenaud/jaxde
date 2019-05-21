from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from jax import jit, vjp
from jax.flatten_util import ravel_pytree
import jax.numpy as np

from jaxde.odeint import odeint

#               ans, [*args               ]
def grad_odeint(yt, func, y0, t, func_args):
    # f(y, t, args)

    T, D = np.shape(yt)
    flat_args, unravel = ravel_pytree(func_args)

    def flat_func(y, t, flat_args):
        return func(y, t, *unravel(flat_args))

    def unpack(x):
        #      y,      vjp_y,      vjp_t,    vjp_args
        return x[0:D], x[D:2 * D], x[2 * D], x[2 * D + 1:]

    @jit
    def augmented_dynamics(augmented_state, t, flat_args):
        # Orginal system augmented with vjp_y, vjp_t and vjp_args.
        y, adjoint, _, _ = unpack(augmented_state)
        dy_dt, vjp_all = vjp(flat_func, y, t, flat_args)
        vjp_a, vjp_t, vjp_args = vjp_all(-adjoint)
        return np.concatenate([dy_dt, vjp_a, vjp_t.reshape(1), vjp_args])

    def vjp_all(g):

        vjp_y = g[-1, :]
        vjp_t0 = 0
        time_vjp_list = []
        vjp_args = np.zeros(np.size(flat_args))

        for i in range(T - 1, 0, -1):

            # Compute effect of moving measurement time.
            vjp_cur_t = np.dot(func(yt[i, :], t[i], *func_args), g[i, :])
            time_vjp_list.append(vjp_cur_t)
            vjp_t0 = vjp_t0 - vjp_cur_t

            # Run augmented system backwards to the previous observation.
            aug_y0 = np.hstack((yt[i, :], vjp_y, vjp_t0, vjp_args))
            aug_ans = odeint(augmented_dynamics, aug_y0, np.stack([t[i], t[i - 1]]), (flat_args,))
            _, vjp_y, vjp_t0, vjp_args = unpack(aug_ans[1])

            # Add gradient from current output.
            vjp_y = vjp_y + g[i - 1, :]

        time_vjp_list.append(vjp_t0)
        vjp_times = np.hstack(time_vjp_list)[::-1]

        return None, vjp_y, vjp_times, unravel(vjp_args)
    return vjp_all
