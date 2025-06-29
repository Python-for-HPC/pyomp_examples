from numba import njit, prange
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_get_wtime
import numpy as np
import math

#
# This program uses a relaxation method to solve a heat diffusion equation.
#  Currently, the results are not validated, though we will add that later.
#

@njit
def initial_value(n, dx, length, u):
    y = dx
    for j in range(n):
        x = dx
        for i in range(n):
            u[j, i] = math.sin(math.pi * x / length) * math.sin(math.pi * y / length)
            x += dx
        y += dx

@njit
def solve(n, alpha, dx, dt, u, u_tmp):
    """Compute the next timestep, given the current timestep"""

    # Finite difference constant multiplier
    r = alpha * dt / (dx ** 2)
    r2 = 1 - 4 * r
    with openmp ("target loop collapse(2)"):
        # Loop over the nxn grid
        for i in range(n):
            for j in range(n):
                u_tmp[j, i] = (r2 * u[j, i] +
                               (u[j, i+1] if i < n-1 else 0.0) +
                               (u[j, i-1] if i > 0   else 0.0) +
                               (u[j+1, i] if j < n-1 else 0.0) +
                               (u[j-1, i] if j > 0 else 0.0))


@njit
def core(nsteps, n, alpha, dx, dt, u, u_tmp):
    with openmp ("target enter data map(to: u, u_tmp)"):
        pass

    for _ in range(nsteps):
        solve(n, alpha, dx, dt, u, u_tmp)
        u, u_tmp = u_tmp, u

    with openmp ("target exit data map(from: u)"):
        pass


n = 10000 # 35000
nsteps = 10
alpha = 0.1
length = 10000
dx = length / (n + 1)
dt = 0.5 / nsteps
u = np.zeros((n,n))
u_tmp = np.zeros((n,n))
initial_value(n, dx, length, u)
nsteps = 10

core(1, n, alpha, dx, dt, u, u_tmp)

u = np.zeros((n,n))
u_tmp = np.zeros((n,n))
initial_value(n, dx, length, u)

tic = omp_get_wtime()
core(nsteps, n, alpha, dx, dt, u, u_tmp)
toc = omp_get_wtime()

print(toc - tic)

