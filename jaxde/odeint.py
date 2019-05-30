from jax.core import Primitive
from jax import jit,jvp,make_jaxpr
from jax.interpreters import xla
from jax import ad

from jaxde.odeint_impl import odeint_impl
from jaxde.ode_jvp import jvp_odeint

def odeint(y0, t, fargs=(), func=None, rtol=1e-7, atol=1e-9, return_evals=False):
    return odeint_p.bind(y0, t, fargs, func, rtol, atol, return_evals)

odeint_p = Primitive('odeint')
odeint_p.def_impl(odeint_impl)
# odeint_p.def_abstract_eval(lambda x:x)
ad.primitive_jvps[odeint_p] = jvp_odeint

