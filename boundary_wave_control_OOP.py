#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 22:29:56 2024

@author: dangthanhvuong
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


class WaveSolver:
    def __init__(self, a, b, T, n, mu, bl, br):
        self.a = a
        self.b = b
        self.T = T
        self.n = n
        self.mu = mu
        self.bl = bl
        self.br = br
        self.dx = (b - a) / (n - 1)
        self.dt = 0.99 * self.dx
        self.m = int(np.ceil(T / self.dt))
        self.x = np.linspace(a, b, n)
        self.t = np.linspace(0, T, self.m)

    @staticmethod
    def u0(x):
        if 1 / 4 < x < 3 / 4:
            return 1
        elif np.isclose(x, 1 / 4) or np.isclose(x, 3 / 4):
            return 0.5
        return 0

    @staticmethod
    def u0t(x):
        return 0

    def wave_direct(self, u0, u0t, mu,bl,br,a,b,T,n):
        u = np.zeros((self.n, self.m))
        s = (self.dt ** 2) / (self.dx ** 2)

        # Initialize the wave
        for i in range(1, self.n - 1):
            u[i, 0] = self.u0(self.x[i])
        u[0, :] = self.bl
        u[self.n - 1, :] = self.br

        # Compute first time step
        for i in range(1, self.n - 1):
            u[i, 1] = u[i, 0] + self.dt * self.u0t(self.x[i])

        # Update the wave over time
        for j in range(2, self.m):
            for i in range(1, self.n - 1):
                u[i, j] = (s * (u[i - 1, j - 1] + u[i + 1, j - 1])
                           + 2 * (1 - s) * u[i, j - 1]
                           - u[i, j - 2])

        return u

    def lambda_function(self, u0, u0t):
        u = self.wave_direct(u0, u0t)
        un = -1 / self.dx * u[self.n - 2, ::-1]
        w = self.wave_direct(np.zeros(self.n - 2), np.zeros(self.n - 2))
        Y = np.concatenate(
            (
                (w[1:self.n - 1, self.m - 2] - w[1:self.n - 1, self.m - 1]) / self.dt,
                -w[1:self.n - 1, self.m - 1]
            ),
            axis=0
        )
        return Y

    def solve(self):
        # Initial conditions for optimization
        u01 = np.array([self.u0(self.x[i + 1]) for i in range(self.n - 2)])
        u01t = np.array([self.u0t(self.x[i + 1]) for i in range(self.n - 2)])
        f = np.concatenate((u01t, -u01))

        # Define the residual function for fsolve
        def residual(X):
            return self.lambda_function(X[:self.n - 2], X[self.n - 2:2 * (self.n - 2)]) - f

        # Solve for Y using fsolve
        Y = fsolve(residual, 2*np.ones_like(f))
        return Y

# Parameters
a = 0
b = 1
T = 3
n = 161
dx = (b - a)/(n - 1)
dt = 0.99*dx
m = np.ceil(T / dt).astype(int)
mu = 1
x = np.linspace(a, b, n)
t = np.linspace(0, T, m)
bl = np.zeros(m)
br = np.zeros(m)

# Create the solver
wave_solver = WaveSolver(a, b, T, n, mu, bl, br)

# Solve the wave equation
Y = wave_solver.solve()

# Compute the direct wave solution
u = wave_solver.wave_direct(Y[:n - 2], Y[n - 2:2 * (n - 2)], mu, bl, br, a, b, T, n)
v = -1 / wave_solver.dx * u[n - 2, :]

# Plot the results
plt.plot(wave_solver.t, v)
plt.legend(['control'])
plt.xlabel('time')
plt.ylabel('value')
plt.title('The discrete control')
plt.show()
