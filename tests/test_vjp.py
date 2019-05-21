from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp

from jaxde.odeint import odeint
from jaxde.ode_vjp import grad_odeint
from jax.flatten_util import ravel_pytree
import jax.numpy as np

def nd(f, x, eps=0.0001):
    flat_x, unravel = ravel_pytree(x)
    D = len(flat_x)
    g = onp.zeros_like(flat_x)
    for i in range(D):
        d = onp.zeros_like(flat_x)
        d[i] = eps
        g[i] = (f(unravel(flat_x + d)) - f(unravel(flat_x - d))) / (2.0 * eps)
    return g


def test_odeint_vjp():
    D = 3
    t0 = 0.1
    t1 = 0.2
    y0 = np.linspace(0.1, 0.9, D)
    fargs = (0.1, 0.2)
    def f(y, t, arg1, arg2):
        return -np.sqrt(t) - y + arg1 - np.mean((y + arg2)**2)

    def onearg_odeint(args):
        return np.sum(odeint(f, *args, atol=1e-8, rtol=1e-8))
    numerical_grad = nd(onearg_odeint, (y0, np.array([t0, t1]), fargs))

    ys = odeint(f, y0, np.array([t0, t1]), fargs, atol=1e-8, rtol=1e-8)
    ode_vjp = grad_odeint(ys, f, y0, np.array([t0, t1]), fargs)
    g = np.ones_like(ys)
    exact_grad, _ = ravel_pytree(ode_vjp(g))

    assert np.allclose(numerical_grad, exact_grad)




