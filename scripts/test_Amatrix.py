"""
Test different a Matrices in 2D

"""

import numpy as np
import matplotlib as mpl

from math import pi


# RK4
dx = 0.1 # TODO maybe change with time

def rk4(dt, x, ds, x0=[0,0], k_f=1):
    x0 =np.array((x0))
    # k1
    xd = ds(x, x0)*k_f
    k1 = dt*xd

    # k2
    xd = ds(x+0.5*k1, x0)*k_f
    k2 = dt*xd

    # k3
    xd = ds(x+0.5*k2, x0)*k_f
    k3 = dt*xd

    # k4
    xd = ds(x+k3, x0)*k_f
    k4 = dt*xd

    # x final
    x = x + 1./6*(k1+2*k2+2*k3+k4) # + O(dt^5)
    
    return x


A = np.array([[-0.1,-1],[1,-1]]) # Unit matrix

# phi = -80/180*pi
# k = -1
# R = np.array([[np.cos(phi), np.sin(phi)],[-np.sin(phi), np.cos(phi)]])
# A = k*R

def ds_int(x, x0=[0,0]):
        return A @ (x-np.array(x0))



# Simualtion parameters
N_int = 1000
N_points = 6
dim = 2

x0 = np.zeros((2,6))
R = 10
dphi = 2*pi/N_points
for ii in range(N_points):
    x0[:,ii] = np.array(([np.cos(dphi*ii)*R,
                          np.sin(dphi*ii)*R]))
dt = 0.05

x = np.zeros((2,N_int+1, N_points))
x[:,0,:] = x0

for nn in range(N_int):
    for ii in range(N_points):
        x[:,nn+1,ii] = rk4(dt, x[:,nn,ii], ds_int)


plt.figure()
plt.plot(x[0,:], x[1,:])

plt.plot(0,0, 'k*', markersize=10)
plt.axis('equal')
plt.ion()
plt.show()

print('\n\n\n... script finished.')
