from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax.numpy as np
from numpy.random import randn
from jax.config import config
config.update("jax_enable_x64", True)

from jax.test_util import check_grads, check_jvp, check_close
from jax import custom_transforms, ad
from jax import jvp, linearize
from jax import make_jaxpr

from jaxde.odeint import odeint
from jaxde.ode_jvp import jvp_odeint

D = 4
t0 = 0.1
t1 = 0.11
ts = np.array([t0, t1])
y0 = np.linspace(0.1, 0.9, D)
fargs = (0.1, 0.2)


def f(y, t, (arg1, arg2)):
    return -np.sqrt(t) - np.sin(np.dot(y, arg1)) - np.mean((y + arg2)**2)

def z(t):
    return odeint(y0,np.array([t0,t]),fargs,func=f)


def fwd_deriv(f):
  def df(t):
    return jvp(f, (t,), (1.0,))[1]
  return df

print(jvp(z, (t0,), (1.0,)))
1/0

g = z
for i in range(5):
    print(g(t0))
    g = fwd_deriv(g)
    # print(g(t0))
    # jaxpr = make_jaxpr(g)(t0)
    # print(jaxpr)  # uncomment to show the blowup
    # print(len(jaxpr.eqns))
