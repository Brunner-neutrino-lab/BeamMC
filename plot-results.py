# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 01:10:22 2022

@author: darro
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import integrate
from scipy import special

def saturation(x, x0, k):
    return x0 * (1 - np.exp(-k * x / x0))

def x0(r):
    a = 50E-3
    return np.pi*r**2/a

def x0_(r):
    a = (50E-3)**2
    return 2*np.pi*r/a

def n0(r,n):
    s = 0.5
    return -n*np.exp(-r**2/2/s**2)

def n0_(r,n):
    s = 0.5
    return n*r/s**2*np.exp(-r**2/2/s**2)

def Nfired_(r,n):
    k = 0.25
    return x0_(r)*(1-np.exp(-k*n0_(r,n)/x0_(r)))

def Nfired_func(n):
    k = 0.25
    a = (50E-3)**2
    s = 0.5
    A = (2*np.pi*s**2/a)
    return A*(np.euler_gamma+np.log(k*n/A)-special.expi(-k*n/A))

Ncells = 100000
PDE = 0.25
N = np.array([1,2,4,8,10,20,40,80,100,200,400,800,1000,2000,4000,8000,10000,20000,40000,80000,100000])
Ngauss = np.array([0.28,0.49,0.96,1.98,2.42,5.24,9.69,20.04,25.11,49.41,95.22,182.81,228.04,417.96,705.60,1096.45,1230.72,1662.68,2099.87,2538.43,2681.9])
Nuni = np.array([0.27,0.55,1.1,1.97,2.49,4.97,10.03,19.71,25.68,50.10,99.92,199.28,247.54,492.01,962.94,1866.06,2287.95,4196.63,7128.95,10620.30,11610.32])
Nslit = np.array([0.25,0.49,1.12,1.89,2.58,5.04,10.18,19.56,24.60,49.73,98.41,194.70,243.48,467.05,892.38,1597.52,1892.32,2995.17,4132.20,5061.56,5310.8])
Nlinear = np.array([0.30,0.51,0.97,2.03,2.52,4.83,9.98,20.34,24.69,49.75,99.54,197.18,244.16,485.56,940.75,1767.71,2151.15,3752.66,5937.55,8292.69,8978.83])

#Ngauss = np.array([0.24,0.48,0.99,2.07,2.46,4.88,9.90,19.98,24.63,48.91,96.48,184.82,227.03,415.99,709.06,1096.01,1235.62,1666.84,2102.63,2541.36])
#Nuni = np.array([0.28,0.51,0.94,2.06,2.44,5.21,10.13,20.33,25.16,49.25,99.19,201.09,249.45,500.31,986.42,1951.99,2426.26,4703.52,8843.50,15728.76])

# Define the figure and axes
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# Plot the data
ax.plot(N[:len(Ngauss)], Ngauss, 'o')
#ax.plot(N[:len(Nuni)], Nuni, 's')
# Plot the saturation curve
x = np.linspace(1, 1E5, int(1E4))
y = saturation(x, Ncells, PDE)
ax.plot(x, y, 'r-')
ax.plot(x, x*PDE, 'k--')
ax.plot(x, Nfired_func(x), '-')

#ax.plot(x, Nfired_func(x), 'g-')
# Set the x and y axes labels
ax.set_xlabel('Number of photons')
ax.set_ylabel('Number of fired cells')
# Set the title
ax.set_title('SiPM saturation curve')
# Set the x and y axis to be logarithmic
ax.set_xscale('log')
ax.set_yscale('log')
# Show the plot
plt.show()

# Fit the data to the saturation curve
#popt, pcov = optimize.curve_fit(saturation, N[:len(Ngauss)], Ngauss)
# Define the figure and axes
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# Plot the saturation curve
r = np.linspace(0,10,num=1000)
for n in N:
    ax.plot(r, Nfired_(r,n), 'k-')
# Set the x and y axes labels
ax.set_xlabel('Radius (mm)')
ax.set_ylabel('Cell firing gradient (1/mm)')
# Set the title
ax.set_title('SiPM Gaussian beam saturation curve')
# Set the x and y scale
ax.set_xlim(0, 3)
ax.set_ylim(0.1, 1E4)
# Set the x and y axis to be logarithmic
#ax.set_xscale('log')
ax.set_yscale('log')
# Show the plot
plt.show()


'''
# Define the figure and axes
fig, ax = plt.subplots(1, 1, dpi=200)
def sat_function(cr,t):
    return cr*t/(1+cr*t)*100
# Plot the sat function
cr = np.logspace(0,6,num=1000)
tmin = -7;tmax = -5
num = tmax-tmin+1 
t = [0.001,0.01,0.1,1,10,100,1000]
for dt in t:
    ax.plot(cr, sat_function(cr,dt*1E-6), '-',label=r'$\Delta t$ = '+str(dt)+r' $\mu s$')
# Set the font size for the axis labels and axis title
ax.tick_params(labelsize=14)
# Set the font size for the x and y labels
ax.xaxis.label.set_size(14)
ax.yaxis.label.set_size(14)
# Set the x and y axes labels
ax.set_xlabel('Source activity [Bq]')
ax.set_ylabel('Percent change in\ndetected avalanches (%)')
# Set the scale to be logarithmic
ax.set_xscale('log')
ax.set_yscale('log')
# Show a grid   
ax.grid(True, which="both", ls="-", color='0.65')
# Show the plot
plt.legend()
plt.tight_layout()
plt.show()
'''