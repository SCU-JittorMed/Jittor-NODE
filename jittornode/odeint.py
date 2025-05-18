import jittor as jt
import jittor.nn as nn
import pdb
from .fixed_grid import Euler, RK4

SOLVERS = {
    'euler': Euler,
    'rk4':RK4,
}

def odeint(func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None):
    options = {} if options is None else options
    solver = SOLVERS[method](func=func, y0=y0, rtol=rtol, atol=atol, **options)
    return solver.integrate(t)