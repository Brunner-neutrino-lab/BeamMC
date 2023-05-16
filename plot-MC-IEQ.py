# -*- coding: utf-8 -*-
"""
Created on Sat May 13 2023

This script plots the results of the Monte Carlo simulation of SiPM saturation
The MC simulation is performed using different photon distributions
The results are compared to the analytical solution of the integral equation

@author: darro
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import integrate
from scipy import special

# Define the integrand for the saturation function
def sat(x,y,A,dist):
    if dist == 'gauss':
        return n*(1-np.exp(-PDE*gauss(A,x,y)/n))
    elif dist == 'uniform':
        return n*(1-np.exp(-PDE*uniform(A)/n))
    elif dist == 'slit':
        return n*(1-np.exp(-PDE*slit(A,x)/n))
    elif dist == 'linear':
        return n*(1-np.exp(-PDE*linear(A,x,y)/n))
    elif dist == 'ellipse':
        return n*(1-np.exp(-PDE*ellipse(A,x,y)/n))
    
# Define the analytical solution of the integral equation
def beam(A,w1,w2):
    # Magic numbers
    s1 = w1/2
    s2 = w2/2
    C = 2*np.pi*s1*s2*n
    return C*(np.euler_gamma+np.log(PDE*A/C)-special.expi(-PDE*A/C))

# Define a function for the gaussian photon distribution
def gauss(A,x,y):
    # Magic numbers
    w1 = 1E-3; s1 = w1/2
    w2 = 1E-3; s2 = w2/2
    return A*np.exp(-(x**2/(2*s1**2)+y**2/(2*s2**2)))/(2*np.pi*s1*s2)

# Define a function for the elliptical photon distribution
def ellipse(A,x,y):
    # Magic numbers
    w1 = 1E-3; s1 = w1/2
    w2 = 1E-3/2; s2 = w2/2
    return A*np.exp(-(x**2/(2*s1**2)+y**2/(2*s2**2)))/(2*np.pi*s1*s2)

# Define a function for the uniform photon distribution
def uniform(A):
    return A/(dx*dy)

# Define a function for the slit photon distribution
def slit(A,x):
    # Magic numbers
    w1 = 1E-3; s1 = w1/2
    return A/dy*np.exp(-(x**2/(2*s1**2)))/(np.sqrt(2*np.pi)*s1)

# Define a function for the linear photon distribution
def linear(A,x,y):
    # Magic numbers
    return A*4/dx/dy*(1/2-x/dx)*(1/2-y/dy)

# Define a function for integrating the 1D saturation function
def IEQ1D(A,dist):
    # Integrate the saturation function from -dx/2 to dx/2 and from -dy/2 to dy/2
    Nava = integrate.quad(sat,-dx/2,dx/2,args=(A,dist))[0]
    return Nava

# Define a function for integrating the 2D saturation function
def IEQ(A,dist):
    # Integrate the saturation function from -dx/2 to dx/2 and from -dy/2 to dy/2
    Nava = integrate.dblquad(sat,-dy/2,dy/2,-dx/2,dx/2,args=(A,dist))[0]
    return Nava

# Magic numbers for the SiPM (HPK VUV4)
# X-dimension of the cell
dx = 5.95E-3
# Y-dimension of the cell
dy = 5.85E-3
# Average PDE of the SiPM
PDE = 0.25
# SiPM cell pitch
cellpitch = 50E-6
# SiPM cell density
n = 1/cellpitch**2

# Setup a dictionary for the photon distributions and avalanches
N = {}
# Setup the names for the photon distributions
dist = ['uniform','gauss','slit','linear','ellipse']

# Initialize the MC results
# Number of photons in the MC distribution
N['phMC'] = np.array([1,2,4,8,10,20,40,80,100,200,400,800,1000,2000,4000,8000,10000,20000,40000,80000,100000])
# Number of avalanches from the MC simulation
N['uniformMC'] = np.array([0.27,0.55,1.1,1.97,2.49,4.97,10.03,19.71,25.68,50.10,99.92,199.28,247.54,492.01,962.94,1866.06,2287.95,4196.63,7128.95,10620.30,11610.32])
N['gaussMC'] = np.array([0.23,0.50,1.03,1.96,2.48,5.09,10.11,19.99,24.75,48.72,96.58,185.24,227.14,417.96,705.60,1096.45,1230.72,1662.68,2099.87,2538.43,2681.9])
N['slitMC'] = np.array([0.25,0.53,1.01,1.97,2.52,4.96,9.86,19.98,24.79,49.80,98.83,195.26,241.92,467.05,892.38,1597.52,1892.32,2995.17,4132.20,5061.56,5310.8])
N['linearMC'] = np.array([0.24,0.47,0.99,2.06,2.44,5.03,9.92,19.82,25.11,49.83,99.40,197.86,245.43,485.56,940.75,1767.71,2151.15,3752.66,5937.55,8292.69,8978.83])
N['ellipseMC'] = np.array([0.27,0.47,1.00,2.08,2.52,4.90,10.00,20.02,23.87,47.43,93.71,172.77,208.51,353.30,549.22,763.47,835.03,1051.60,1269.03,1487.16,1560.63])

# Initialize the analytical results
# Number of photons in the IEQ distribution
N['ph'] = np.logspace(0,np.log10(max(N['phMC'])),100)

# Number of avalanches from an ideal SiPM
N['ideal'] = N['ph']*PDE

for d in dist:
    figure, axes = plt.subplots(2,2)
    N[d] = []
    for ph in N['ph']:
        N[d].append(IEQ(ph,d))

    for ax in axes:
        for a in ax:
            # Plot the MC results
            a.plot(N['phMC'],N['%sMC'%d],'o',label=d+' MC')
            # Plot the ideal response
            a.plot(N['ph'],N['ideal'],'--',label='Ideal response')
            # Plot the IEQ results
            a.plot(N['ph'],N[d],'-',label=d+' IEQ')

            if d == 'ellipse':
                # Plot the beam profile
                a.plot(N['ph'],beam(N['ph'],1E-3,0.5E-3),'-.',label='Analytical solution')
            if d == 'gauss':
                # Plot the beam profile
                a.plot(N['ph'],beam(N['ph'],1E-3,1E-3),':',label='Analytical solution')

            # Set the axes labels
            a.set_xlabel('Number of photons')
            a.set_ylabel('Number of avalanches')
            # Set the legend
            a.legend()

    # Set the axes scale
    axes[0][1].set_xscale('log')

    axes[1][0].set_yscale('log')

    axes[1][1].set_xscale('log')
    axes[1][1].set_yscale('log')

    figure.tight_layout()
    # Show the plot
    plt.show()
