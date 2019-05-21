from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from jaxde.odeint import odeint
import jax.numpy as np

def test_fwd_back():
    # Run a system forwards then backwards,
    # and check that we end up in the same place.
    D = 10
    t0 = 0.1
    t1 = 2.2
    y0 = np.linspace(0.1, 0.9, D)

    def f(y, t):
        return -np.sqrt(t) - y + 0.1 - np.mean((y + 0.2)**2)

    ys  = odeint(f, y0,     np.array([t0, t1]), atol=1e-8, rtol=1e-8)
    rys = odeint(f, ys[-1], np.array([t1, t0]), atol=1e-8, rtol=1e-8)

    assert np.allclose(y0, rys[-1])


