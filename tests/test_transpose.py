from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax.numpy as np
from numpy.random import randn
from jax.config import config
config.update("jax_enable_x64", True)

from jax.test_util import check_grads, check_jvp, check_close
from jax import custom_transforms, ad
from jax import jvp,linearize
from jax import make_jaxpr

from jaxde.odeint import odeint
from jaxde.ode_jvp import jvp_odeint

# Parameters for the test function
#TODO: Test more functions
D = 4
t0 = 0.1
t1 = 0.11
ts= np.array([t0,t1])
y0 = np.linspace(0.1, 0.9, D)
fargs = (0.1, 0.2)


def f(y, t, arg1, arg2):
    return -np.sqrt(t) - np.sin(np.dot(y, arg1)) - np.mean((y + arg2)**2)

def test_odeint_2_linearize():

    def odeint2(y0, ts, fargs):
        return odeint(f, y0, ts, fargs, atol=1e-8,
                      rtol=1e-8)
    odeint2_prim = custom_transforms(odeint2).primitive

    def odeint2_jvp((y0, ts, fargs), (tan_y, tan_ts, tan_fargs)):
        return jvp_odeint(f, (y0, ts, fargs),
                          (tan_y, tan_ts, tan_fargs))
    ad.defjvp(odeint2_prim,odeint2_jvp)

    _, out_tangent = jvp(odeint2,(y0,ts,fargs), (y0,ts,fargs)) # when break this is why
    y, f_jvp = linearize(odeint2,*(y0,ts,fargs))
    out_tangent_2 = f_jvp(*(y0,ts,fargs))
    
    # print(make_jaxpr(f_jvp)(y0,t0,t1,fargs))
    check_close(out_tangent,out_tangent_2)


def test_odeint_linearize():

    _, out_tangent = jvp(odeint,(f,y0,ts,fargs), (None,y0,ts,fargs)) # when break this is why
    y, f_jvp = linearize(odeint,*(f,y0,ts,fargs))
    out_tangent_2 = f_jvp(*(f,y0,ts,fargs))
    
    # print(make_jaxpr(f_jvp)((y0,t0,t1,fargs),))
    check_close(out_tangent,out_tangent_2)
