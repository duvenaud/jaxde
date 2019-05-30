from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax.numpy as np
from numpy.random import randn
from jax.config import config
config.update("jax_enable_x64", True)

from jax.test_util import check_grads, check_jvp
from jax import custom_transforms, ad
from jax import jvp
from functools import partial

from jaxde.odeint import odeint
# from jaxde.ode_jvp import jvp_odeint

import pdb

# Parameters for the test function
#TODO: Test more functions
D = 4
t0 = 0.1
t1 = 0.11
ts = np.array([t0,t1])
y0 = np.linspace(0.1, 0.9, D)
fargs = (0.1, 0.2)


def f(y, t, (arg1, arg2)):
    return -np.sqrt(t) - np.sin(np.dot(y, arg1)) - np.mean((y + arg2)**2)


def test_odeint_jvp():
    def odeint2(y0, ts, fargs):
        return odeint(y0,
                      ts,
                      fargs,
                      func=f,
                      atol=1e-8,
                      rtol=1e-8)

    def odeint2_jvp((y0, ts, fargs), (tan_y, tan_ts, tan_fargs)):
        jvp_odeint = partial(jvp,odeint2)
        return jvp_odeint((y0, ts, fargs),
                          (tan_y, tan_ts, tan_fargs))

    check_jvp(odeint2, odeint2_jvp, (y0, ts, fargs))
