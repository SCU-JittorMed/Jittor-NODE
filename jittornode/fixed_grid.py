import jittor as jt
import jittor.nn as nn
import pdb
import abc

def is_close(a, b, tol=1e-6):
    return (jt.abs(a - b) < tol)

class FixedGridODESolver(metaclass=abc.ABCMeta):
    order: int

    def __init__(self, func, y0, step_size=None, grid_constructor=None, interp="linear", **unused_kwargs):
        self.func = func
        self.y0 = y0
        self.dtype = y0.dtype
        self.step_size = step_size
        self.interp = interp

        if step_size is None:
            if grid_constructor is None:
                self.grid_constructor = lambda f, y0, t: t
            else:
                self.grid_constructor = grid_constructor
        else:
            if grid_constructor is None:
                self.grid_constructor = self._grid_constructor_from_step_size(step_size)
            else:
                raise ValueError("step_size and grid_constructor are mutually exclusive arguments.")

    @staticmethod
    def _grid_constructor_from_step_size(step_size):
        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]
            niters = int(jt.ceil((end_time - start_time) / step_size + 1).item())
            t_infer = jt.arange(0, niters, dtype=t.dtype) * step_size + start_time
            t_infer[-1] = t[-1]
            return t_infer
        return _grid_constructor

    @abc.abstractmethod
    def _step_func(self, func, t0, dt, t1, y0):
        pass

    def integrate(self, t):
        time_grid = self.grid_constructor(self.func, self.y0, t)
        
        if not (is_close(time_grid[0], t[0]) and is_close(time_grid[-1], t[-1])):
            raise ValueError("time_grid does not match given t")

        solution = []
        solution.append(self.y0)

        y0 = self.y0
        j = 1
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dt = t1 - t0
            dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy

            while j < len(t) and t1 >= t[j]:
                if self.interp == "linear":
                    sol_j = self._linear_interp(t0, t1, y0, y1, t[j])
                else:
                    sol_j = y1
                solution.append(sol_j)
                j += 1
            y0 = y1

        return jt.stack(solution)

    def _linear_interp(self, t0, t1, y0, y1, t):
        if is_close(t, t0):
            return y0
        if is_close(t, t1):
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)


class Euler(FixedGridODESolver):
    order = 1
    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0)
        return dt * f0, f0


def rk4_alt_step_func(func, t0, dt, t1, y0, f0=None):
    """Smaller error with slightly more compute."""
    # Precompute divisions
    _one_third = 1 / 3
    _two_thirds = 2 / 3
    _one_sixth = 1 / 6 
    
    k1 = f0
    if k1 is None:
        k1 = func(t0, y0, perturb=Perturb.NEXT if perturb else Perturb.NONE)
    k2 = func(t0 + dt * _one_third, y0 + dt * k1 * _one_third)
    k3 = func(t0 + dt * _two_thirds, y0 + dt * (k2 - k1 * _one_third))
    k4 = func(t1, y0 + dt * (k1 - k2 + k3))
    
    return (k1 + 3 * (k2 + k3) + k4) * dt * 0.125


class RK4(FixedGridODESolver):
    order = 4

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0)
        return rk4_alt_step_func(func, t0, dt, t1, y0, f0=f0), f0
    