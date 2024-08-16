==#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 22:29:56 2024

@author: dangthanhvuong
"""
import numpy as np
import matplotlib.pyplot as plt
from math import sin
from scipy.optimize import fsolve
# Define functions u0 and u0t if needed
def u0(x):
    if(1/4 < x <3/4):
        return 1
    if(x == 1/4):
        return 0.5
    if(x == 3/4):
        return 0.5
    if(x<1/4):
        return 0
    if(x>3/4):
        return 0
def u0t(x):
    return 0

def WaveDir(u0, u0t, mu, bl, br, a, b, T, n, m):
    u = np.zeros((n, m))
    dx = (b - a) / (n - 1)
    dt = T / (m - 1)
    x = np.linspace(a, b, n)
    t = np.linspace(0, T, m)
    s = (dt ** 2) / (dx ** 2)

    # Create the matrix A
    e = np.ones(n - 2)
    e1 = np.ones(n - 3)
    A = np.diag(s*e1, -1) + np.diag(2*(1 - s)*e, 0) + np.diag(s*e1, 1)
    #A = A[:-1, :-1]  # remove last row and column to match MATLAB's indexing
    for i in range(1,n-1):
        if(1/4 < x[i] <3/4):
            u[i,0] = 1
        if(x[i] == 1/4):
            u[i,0] = 0.5
        if(x[i] == 3/4):
            u[i,0] = 0.5
        if(x[i]<1/4):
            u[i,0] = 0        
        if(x[i]>3/4):
            u[i,0] = 0  # use initial conditions to compute first two points
    u[0,:] = bl  # Point 0
    u[n-1,:] = br  # Point 1
    for i in range(n-2):
        u[i + 1, 1] = u[i + 1, 0] + dt*0

    # Update the boundary point
    I = np.zeros(n - 2)
    for j in range(2, m):
        I[0] = s*u[0, j - 1]
        I[n - 3] = s*u[n - 1, j - 1]
        u[1:n-1, j] = np.dot(A, u[1:n - 1, j - 1]) - u[1:n - 1, j - 2] + I

    return u
def Lambda(u0,u0t,mu,bl,br,a,b,T,n,m):
    u = WaveDir(u0,u0t,mu,bl,br,a,b,T,n,m)
    dx = (b-a)/(n-1)
    x = np.transpose(np.arange(a,b+dx,dx))
    dt = T/(m-1)
    t = np.arange(0,T+dt,dt)
    un = np.zeros((1,m))
    for i in range(m):
        un[0,i] = -1/dx*u[n-2,m-1-i]
    w01 = np.zeros((n-2,1))
    w01t = np.zeros((n-2,1))
    w = WaveDir(w01,w01t,mu,bl,un,a,b,T,n,m) #backward problem
    Y = np.concatenate((np.dot(1/dt,(w[1:n-1,m-2] - w[1:n-1,m-1])),np.dot(-1,w[1:n-1,m-1])),axis=0)
    return Y
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

# Compute the solution by given data
u01 = np.zeros(n - 2)
u01t = np.zeros(n - 2)
for i in range(n - 2):
    u01[i] = u0(x[i + 1])
    u01t[i] = u0t(x[i + 1])
print(u01)
print(u01t)
f = np.concatenate((u01t, -u01))
bfun = lambda X: Lambda(X[:n-2], X[n-2:2*(n-2)], mu, bl, br, a, b, T, n, m) - f
Y = fsolve(bfun, 2*np.ones_like(f))
bl = np.zeros(m)
br = np.zeros(m)
u = WaveDir(Y[:n - 2], Y[n - 2:2*(n - 2)], mu, bl, br, a, b, T, n, m)
v = -1/dx*u[n-2,:]
plt.plot(t, v)
plt.legend(['control'])
plt.xlabel('time')
plt.ylabel('value')
plt.title('The discrete control')
plt.show()