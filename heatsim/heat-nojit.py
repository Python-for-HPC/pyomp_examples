from numba.openmp import omp_get_wtime
import numpy as np
import sys
import math

def initial_value(n, dx, length, u):
    y = dx
    for j in range(n):
        x = dx
        for i in range(n):
            u[j, i] = math.sin(math.pi * x / length) * math.sin(math.pi * y / length)
            x += dx
        y += dx

def solution(t, x, y, alpha, length):
    return math.exp(-2.0 * alpha * (math.pi ** 2) * t / (length ** 2)) * math.sin(math.pi * x / length) * math.sin(math.pi * y / length)


def l2norm(n, u, nsteps, dt, alpha, dx, length):
    time = dt * nsteps
    l2norm_ret = 0.0

    y = dx
    for j in range(n):
        x = dx
        for i in range(n):
            answer = solution(time, x, y, alpha, length)
            l2norm_ret += (u[j, i] - answer) ** 2
            x += dx
        y += dx

    return math.sqrt(l2norm_ret)


def solve(n, alpha, dx, dt, u, u_tmp):
    r = alpha * dt / (dx ** 2)
    r2 = 1.0 - 4.0 * r
    for i in range(n):
        for j in range(n):
            u_tmp[j, i] = (r2 * u[j, i] +
                           (r * u[j, i+1] if i < n-1 else 0.0) +
                           (r * u[j, i-1] if i > 0   else 0.0) +
                           (r * u[j+1, i] if j < n-1 else 0.0) +
                           (r * u[j-1, i] if j > 0 else 0.0))

def core(nsteps, n, alpha, dx, dt, u, u_tmp):
    for t in range(nsteps):
        solve(n, alpha, dx, dt, u, u_tmp)
        u, u_tmp = u_tmp, u

if __name__ == "__main__":
    start = omp_get_wtime()

    n = 1000
    nsteps = 10

    if len(sys.argv) == 3:
        n = int(sys.argv[1])
        nsteps = int(sys.argv[2])

    alpha = 0.1
    length = 1000.0
    dx = length / (n + 1)
    dt = 0.5 / nsteps
    r = alpha * dt / (dx ** 2)
    print(" MMS heat equation")
    print("Problem input")
    print(f" Grid size: {n} x {n}")
    print(f" Cell width: {dx}")
    print(f" Grid length: {length} x {length}\n")
    print(f" Alpha: {alpha}\n");
    print(f" Steps: {nsteps}");
    print(f" Total time: {dt*nsteps}");
    print(f" Time step: {dt}");

    # Stability check
    print("Stability");
    print(f" r value: {r}");
    if r > 0.5:
        print("Warning: unstable")

    u = np.zeros((n,n))
    u_tmp = np.zeros((n,n))
    initial_value(n, dx, length, u)

    tic = omp_get_wtime()
    core(nsteps, n, alpha, dx, dt, u, u_tmp)
    toc = omp_get_wtime()

    norm = l2norm(n, u, nsteps, dt, alpha, dx, length)

    stop = omp_get_wtime()

    print("Error (L2norm):", norm)
    print("Solve time (s):", toc-tic)
    print("total time:", stop-start)

print("DONE")
