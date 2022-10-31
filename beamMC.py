# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 01:10:22 2022

@author: darro
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import time
import os

# Create an array of photons with a gaussian beam profile
def MCbeam():
    X=[];Y=[]
    for i in range(N):
        X.append(random.gauss(0,w1/2))
        Y.append(random.gauss(0,w2/2))
    data['hitx'] = X
    data['hity'] = Y

# Define the bounds for the indexed SPAD
def cell(i,ds):
    x1=i*pitch-ds/2
    x2=(i+1)*pitch-ds/2
    return x1,x2

def raster(t0,xs,xe,ys,ye):
    # Iterate through x-SPADs
    for i in range(xs,xe):
        x1,x2 = cell(i,dx)
        # Iterate through y-SPADs
        for j in range(ys,ye):
            # Has this SPAD already been checked
            if data['sipm'][i][j] == 0:
                continue
            # Are there any photons remaining
            if len(data['hitx']) == 0:
                continue  
            y1,y2 = cell(j,dy)   
            # Initialize number of hits and the hit array for the indexed SPAD
            n=0;rm=[]
            # Iterate through photon array, checking if they hit the indexed SPAD
            for k in range(len(data['hitx'])):
                if x1 < data['hitx'][k]:
                    if x2 > data['hitx'][k]:
                        if y1 < data['hity'][k]:
                            if y2 > data['hity'][k]:
                                # If the photon hits the indexed SPAD, count it and add the photon index to the hit array
                                n+=1;rm.append([data['hitx'][k],data['hity'][k]])
            # Count the total number of hits for the indexed SPAD
            data['x%i_y%i'%(i,j)] = n
            # Remove photons that have been counted from the photon array
            for ph in rm:
                data['hitx'].remove(ph[0])
                data['hity'].remove(ph[1])
            # Remove the SPAD from the SPAD array
            data['sipm'][i][j] = 0

# Determine how many photons hit each SPAD    
def MCSiPM(pde):
    # Initialize SPAD index, time, and the SPAD-hit array
    T0=time.time();SPADinit()  
    # Create an array of physical coordinates
    x=[];y=[]
    for i in range(xi):
        x.append(cell(i,dx)[0])
    for i in range(yi):
        y.append(cell(i,dy)[0])   
    x=np.array(x);y=np.array(y)
    # Compare the physical coordinates against the beam radius
    # Determine the SPAD indicies within each box of radius i*w0/2
    rx=[];ry=[]
    for i in range(1,5):
        rx.append([np.where(x<-w1*i/2)[0][-1],np.where(x<w1*i/2)[0][-1]])
        ry.append([np.where(y<-w2*i/2)[0][-1],np.where(y<w2*i/2)[0][-1]])
    rx.append([0,xi]);ry.append([0,yi])
    # Array for tracking which SPAD has been checked
    data['sipm'] = np.ones([xi,yi])
    # Raster boxes with increasing size about the center of the beam - this increases the speed of the process
    for i in range(len(rx)):
        raster(T0,rx[i][0],rx[i][1],ry[i][0],ry[i][1]) 
    # Simulate firing SPADs
    data['ava'] = []
    for i in range(xi):
        for j in range(yi):
            data['x%i_y%i_ava'%(i,j)] = 0
            for k in range(data['x%i_y%i'%(i,j)]):
                if random.random() < pde:
                    data['x%i_y%i_ava'%(i,j)] += 1
            data['ava'].append(data['x%i_y%i_ava'%(i,j)])

def SPADinit():
    for i in range(xi):
        for j in range(yi):     
            # Initialize the SPAD hit array
            data['x%i_y%i'%(i,j)] = 0
            
# Determine the SPAD-hit distribution
def MChit():
    Nhit = 0;Nmult = 0;N1 = 0;N0 = 0
    for i in range(xi):
        for j in range(yi):
            Nhit += data['x%i_y%i'%(i,j)]
            if data['x%i_y%i'%(i,j)] == 0:
                N0 += 1
            elif data['x%i_y%i'%(i,j)] == 1:
                N1 += 1
            elif data['x%i_y%i'%(i,j)] > 1:
                Nmult += 1
            else:
                print('except %s'%data['x%i_y%i'%(i,j)])
    return Nhit,Nmult,N1,N0

# Determine the SPAD-detection distribution
def MCdetect():
    Nhit = 0;Nmiss = 0;N1 = 0;N0 = 0
    for i in range(xi):
        for j in range(yi):
            if data['x%i_y%i_ava'%(i,j)] > 0:
                Nhit += 1
            else:
                N0 += 1
            if data['x%i_y%i_ava'%(i,j)] > 1:
                Nmiss += data['x%i_y%i_ava'%(i,j)]-1
    return Nhit,Nmiss,N1,N0

def updatesim(i,t0,pde,sim,z):
    if i == 0:
        if z == 0:
            print('Simulation beginning\n')
    elif i % 10 == 0:
        print('Status: PDE = %.2f %%'%(pde*100))
        # Check how much time is remaining
        pc_part = sim/(i)
        t=time.time()-t0
        tf = t*(pc_part-1)/60
        print('%i minutes remaining on this PDE'%(tf))
        wh_part = PDEs/(z+1/pc_part)
        t=time.time()-start
        tf = t*(wh_part-1)/60
        print('%i minutes remaining on the simulation\n'%(tf))

# Guassian beam plot
def plotflux(i):
    plots['fig_flux'],plots['ax_flux']=plt.subplots(1,1,dpi=100)
    plots['ax_flux'].hist2d(data['hitx'],data['hity'],bins=(xi,yi),range=[[-dx/2,dx/2],[-dx/2,dx/2]])
    plots['ax_flux'].set_xlabel('X position [mm]')
    plots['ax_flux'].set_ylabel('Y position [mm]')
    plots['ax_flux'].axis('square')
    plots['ax_flux'].axis('square')
    plots['fig_flux'].savefig('plots/simulation_%i.png'%i)
    plots['fig_flux1d'],plots['ax_flux1d']=plt.subplots(2,1,dpi=100)
    plots['ax_flux1d'][0].hist(data['hitx'],bins=xi)
    plots['ax_flux1d'][1].hist(data['hity'],bins=yi)
    plots['ax_flux1d'][0].set_xlabel('X position [mm]')
    plots['ax_flux1d'][1].set_xlabel('Y position [mm]')
    plots['ax_flux1d'][0].set_ylabel('SPAD hits')
    plots['ax_flux1d'][1].set_ylabel('SPAD hits')
    plots['fig_flux1d'].savefig('plots/simulation1d_%i.png'%i)
    plt.close()

# SPAD-hit distribution    
def plothits():
    hits=[]
    for i in range(xi):
        for j in range(yi):
            hits.append(data['x%i_y%i'%(i,j)])
    plots['fig_hits'],plots['ax_hits']=plt.subplots(1,1,dpi=100)
    plots['ax_hits'].hist(hits,bins=np.arange(20))
    plots['ax_hits'].set_xlabel('Number of photons hitting a single SPAD')
    plots['ax_hits'].set_ylabel('SPAD hit frequency')
    plots['ax_hits'].set_yscale('log')
    plt.close()

# SPAD-hit distribution    
def plotava():
    hits=[]
    for i in range(xi):
        for j in range(yi):
            hits.append(data['x%i_y%i_ava'%(i,j)])
    plots['fig_ava'],plots['ax_ava']=plt.subplots(1,1,dpi=100)
    plots['ax_ava'].hist(hits,bins=np.arange(10))
    plots['ax_ava'].set_xlabel('Number of photon-induced avalanches')
    plots['ax_ava'].set_ylabel('SPAD avalanche frequency')
    plots['ax_ava'].set_yscale('log')
    plt.close()
    
# Measured and missing avalanches
def plotresults(pde):
    mc=np.average(data['mc']);mcs=np.std(data['mc'])/np.sqrt(len(data['mc']))
    xc=np.average(data['xc']);xcs=np.std(data['xc'])/np.sqrt(len(data['xc']))
    fig,ax=plt.subplots(2,1,dpi=100,figsize=(8,8))
    ax[0].hist(data['mc'],label='%.2f +/- %.2f'%(mc,mcs))         
    ax[1].hist(data['xc'],label='%.2f +/- %.2f'%(xc,xcs))
    ax[0].set_xlabel('Measured avalanches')
    ax[0].set_ylabel('MC frequency')
    ax[1].set_xlabel('Missing avalanches')
    ax[1].set_ylabel('MC frequency')
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    fig.savefig('plots/results_PDE%i.png'%(100*pde))
    plt.close()
   
def main(pde,z):   
    # Main function
    # Repeat measurement
    data['mc'],data['xc']=[],[]
    sims=10;t0=time.time()
    for i in range(sims):
        updatesim(i,t0,pde,sims,z)
        # MC Gaussian beam
        MCbeam()
        # Plot the Gaussian beam
        #plotflux(i)  
        # Determine how many photons hit each SPAD
        MCSiPM(pde) 
        # Determine the SPAD-hit distribution
        Hhit,Hmiss,H1,H0=MChit()
        # Plot a histogram of the SPAD-hit distribution
        #plothits()
        # Determine how many SPADs fire
        Ahit,Amiss,A1,A0=MCdetect()
        # Plot a histogram of the SPAD-avalanche distribution
        #plotava()    
        data['mc'].append(Ahit)
        data['xc'].append(Amiss)
    # Plots the results of the simulation for the current PDE    
    plotresults(pde)
    
if __name__ == "__main__":  
    start=time.time()
    
    # Magic numbers 
    # Beam profile       
    # How many photons are in the pulse     
    N = int(6.761728E6/500)
    # What is the beam radius (Gaussian beam), (x,y)
    w1=1.29;w2=1.31#mm
    # SiPM/SPAD parameters
    PDE = np.arange(1,30)/100
    PDEs = len(PDE)
    pitch = 50E-3#mm
    # SiPM width/height
    dx = 5.95
    dy = 5.85
    
    # SPAD array
    xi=int(dx/pitch)
    yi=int(dy/pitch)
      
    # Make dictionaries
    plots = {}
    data = {}
    # Make directories
    os.mkdir('plots')
    
    # Main function
    for i in range(PDEs):
        main(PDE[i],i)
 
    end=time.time()
    print('Simulation complete - time elapsed: %.2f hours'%(end-start)/3600)
































