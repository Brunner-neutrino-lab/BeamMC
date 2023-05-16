# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 01:10:22 2022

@author: darro
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
import os

# Create an array of photons with a gaussian beam profile
def MCbeam(n):
    X=[];Y=[]
    for i in range(n):
        X.append(random.gauss(0,w1/2))
        Y.append(random.gauss(0,w2/2))
    data['hitx'] = X
    data['hity'] = Y

# Create an array of photons with one-dimension uniform and the other gaussian
def MCslit(n):
    X=[];Y=[]
    for i in range(n):
        X.append(random.gauss(0,w1/2))
        Y.append(random.uniform(-dy/2,dy/2))
    data['hitx'] = X
    data['hity'] = Y    

# Create an array of photons with a uniform beam profile
def MCuni(n):
    X=[];Y=[]
    for i in range(n):
        X.append(random.uniform(-dx/2,dx/2))
        Y.append(random.uniform(-dy/2,dy/2))
    data['hitx'] = X
    data['hity'] = Y    

# Create an array of photons with a uniform beam profile
def MClinear(n):
    X=[];Y=[]
    for i in range(n):
        X.append(random.triangular(-dx/2,dx/2,-dx/2))
        Y.append(random.triangular(-dy/2,dy/2,-dy/2))
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
            nhit=0;rm=[]
            # Iterate through photon array, checking if they hit the indexed SPAD
            for k in range(len(data['hitx'])):
                if x1 < data['hitx'][k]:
                    if x2 > data['hitx'][k]:
                        if y1 < data['hity'][k]:
                            if y2 > data['hity'][k]:
                                # If the photon hits the indexed SPAD, count it and add the photon index to the hit array
                                nhit+=1;rm.append([data['hitx'][k],data['hity'][k]])
            # Count the total number of hits for the indexed SPAD
            data['x%i_y%i'%(i,j)] = nhit
            data['hits'].append(nhit)
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
    if SIM == 'gauss' or SIM == 'agauss' or SIM == 'slit':     
        if w1 < dx and w2 < dy:
            for i in range(1,5):
                rx.append([np.where(x<-w1*i/2)[0][-1],np.where(x<w1*i/2)[0][-1]])
                ry.append([np.where(y<-w2*i/2)[0][-1],np.where(y<w2*i/2)[0][-1]])
    rx.append([0,xi]);ry.append([0,yi])
    # Array for tracking which SPAD has been checked
    data['sipm'] = np.ones([xi,yi])
    # Flat array for tracking the number of hits
    data['hits'] = []
    # Raster boxes with increasing size about the center of the beam - this increases the speed of the process
    for i in range(len(rx)):
        raster(T0,rx[i][0],rx[i][1],ry[i][0],ry[i][1]) 
    # Flat array for tracking SPADs that should avalanche
    data['ava'] = []
    for i in range(xi):
        for j in range(yi):
            data['x%i_y%i_ava'%(i,j)] = 0
            for k in range(data['x%i_y%i'%(i,j)]):
                if random.random() < pde:
                    data['x%i_y%i_ava'%(i,j)] += 1
            data['ava'].append(data['x%i_y%i_ava'%(i,j)])

def SPADinit():
    # Loop through the SPAD indicies
    for i in range(xi):
        for j in range(yi):     
            # Initialize the SPAD hit array
            data['x%i_y%i'%(i,j)] = 0
            
# Determine the SPAD-hit distribution
def MChit():
    Nhit = 0;Nmult = 0;N1 = 0;N0 = 0
    # Loop through the SPAD indicies
    for i in range(xi):
        for j in range(yi):
            # Count the number of hits, multiple hits, and single hits
            Nhit += data['x%i_y%i'%(i,j)]
            if data['x%i_y%i'%(i,j)] == 0:
                N0 += 1
            elif data['x%i_y%i'%(i,j)] == 1:
                N1 += 1
            elif data['x%i_y%i'%(i,j)] > 1:
                Nmult += 1
            else:
                # If the number of hits is not an expected result, print the error
                print('except %s'%data['x%i_y%i'%(i,j)])
    # Return the number of hits, multiple hits, and single hits
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

def updatesim(s,t0,pde,sim,p):
    if s == 0:
        if p == 0:
            print('Simulation beginning\n')
    # Print the percentage of simulation processed, updating every 10%
    elif s % (sim/10) == 0:
    #elif s % 10 == 0:
        print('Status: PDE = %.2f %% - %i %% complete'%(pde*100,s/sim*100))
        # Check how much time is remaining
        pc_part = sim/(s)
        t=time.time()-t0
        tf = t*(pc_part-1)/60
        print('%i minutes remaining on this PDE'%(tf))
        wh_part = PDEs/(p+1/pc_part)
        t=time.time()-start
        tf = t*(wh_part-1)/60
        print('%i minutes remaining on the simulation\n'%(tf))

# Guassian beam plot
def plotflux(s,n):
    if s == 0:
        # Plot the flux
        plots['fig_flux'],plots['ax_flux']=plt.subplots(1,1,dpi=100)
        plots['ax_flux'].hist2d(data['hitx'],data['hity'],bins=(xi,yi),range=[[-dx/2,dx/2],[-dx/2,dx/2]])
        #plots['ax_flux'].colorbar()
        plots['ax_flux'].set_xlabel('X position [mm]')
        plots['ax_flux'].set_ylabel('Y position [mm]')
        plots['ax_flux'].axis('square')
        plots['ax_flux'].axis('square')
        plt.tight_layout()
        plots['fig_flux'].savefig('plots/%s/sim-flux_%i_Nph_%i.png'%(SIM,s,n))
        plt.close(plots['fig_flux'])
        # Plot the flux in 1D
        plots['fig_flux1d'],plots['ax_flux1d']=plt.subplots(2,1,dpi=100)
        plots['ax_flux1d'][0].hist(data['hitx'],bins=xi,range=[-dx/2,dx/2])
        plots['ax_flux1d'][1].hist(data['hity'],bins=yi,range=[-dy/2,dy/2])
        plots['ax_flux1d'][0].set_xlabel('X position [mm]')
        plots['ax_flux1d'][1].set_xlabel('Y position [mm]')
        plots['ax_flux1d'][0].set_ylabel('SPAD hits')
        plots['ax_flux1d'][1].set_ylabel('SPAD hits')
        plt.tight_layout()
        plots['fig_flux1d'].savefig('plots/%s/sim-flux1d_%i_Nph_%i.png'%(SIM,s,n))
        #plt.close(plots['fig_flux1d'])

# SPAD-hit distribution    
def plothits(s,n):
    if s == 0:
        # Plot the SPAD-hit distribution
        plots['fig_hits'],plots['ax_hits']=plt.subplots(1,1,dpi=100)
        plots['ax_hits'].hist(data['hits'],bins=np.arange(int(max(data['hits']))+2),align='left')
        plots['ax_hits'].set_xlabel('Number of photons hitting a single SPAD')
        plots['ax_hits'].set_ylabel('SPAD hit frequency')
        plots['ax_hits'].set_yscale('log')
        plots['ax_hits'].xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plots['fig_hits'].savefig('plots/%s/sim-hit-density_%i_Nph_%i.png'%(SIM,s,n))
        plt.close(plots['fig_hits'])

# SPAD-avalanche distribution    
def plotava(s,n):
    if s == 0:
        # Plot the SPAD-avalanche distribution
        plots['fig_ava'],plots['ax_ava']=plt.subplots(1,1,dpi=100)
        plots['ax_ava'].hist(data['ava'],bins=np.arange(int(max(data['ava']))+2),align='left')
        plots['ax_ava'].set_xlabel('Number of photon-induced avalanches')
        plots['ax_ava'].set_ylabel('SPAD avalanche frequency')
        plots['ax_ava'].set_yscale('log')
        plots['ax_ava'].xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plots['fig_ava'].savefig('plots/%s/sim-avalanche-density_%i_Nph_%i.png'%(SIM,s,n))
        plt.close(plots['fig_ava'])
    
# Measured and missing avalanches
def plotresults(pde,n):
    mc=np.average(data['mc']);mcs=np.std(data['mc'])/np.sqrt(len(data['mc'])-1)
    xc=np.average(data['xc']);xcs=np.std(data['xc'])/np.sqrt(len(data['xc'])-1)
    fig,ax=plt.subplots(2,1,dpi=100,figsize=(8,8))
    param_str = "PDE = %.2f %%\nN = %i\nN_cells = %i\nSimulation = %s\n"%(pde*100,n,Ncells,SIM)
    ax[0].hist(data['mc'],label=param_str+'%.2f +/- %.2f'%(mc,mcs))         
    ax[1].hist(data['xc'],label=param_str+'%.2f +/- %.2f'%(xc,xcs))
    ax[0].set_xlabel('Measured avalanches')
    ax[0].set_ylabel('MC frequency')
    ax[1].set_xlabel('Missing avalanches')
    ax[1].set_ylabel('MC frequency')
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    fig.savefig('plots/%s/results_PDE_%i_Nph_%i_Ncells_%i.png'%(SIM,100*pde,n,Ncells))
    plt.close(fig)
   
def main(pde,p,n):   
    # Main function
    # Repeat measurement
    data['mc'],data['xc']=[],[]
    sims=100;t0=time.time()
    for s in range(sims):
        updatesim(s,t0,pde,sims,p)
        # MC beam
        if SIM == 'gauss':
            MCbeam(n)
        if SIM == 'ellipse':
            MCbeam(n)
        if SIM == 'slit':
            MCslit(n)
        if SIM == 'uni':
            MCuni(n)
        if SIM == 'linear':
            MClinear(n)
        # Plot the beam
        plotflux(s,n)  
        # Determine how many photons hit each SPAD
        MCSiPM(pde) 
        # Determine the SPAD-hit distribution
        Hhit,Hmiss,H1,H0=MChit()
        # Plot a histogram of the SPAD-hit distribution
        plothits(s,n)
        # Determine how many SPADs fire
        Ahit,Amiss,A1,A0=MCdetect()
        # Plot a histogram of the SPAD-avalanche distribution
        plotava(s,n)    
        data['mc'].append(Ahit)
        data['xc'].append(Amiss)
    # Plots the results of the simulation for the current PDE    
    plotresults(pde,n)
    # Save the data

    
if __name__ == "__main__":  
    # Start the timer
    start=time.time()
    
    # Magic numbers 
    # Beam profile       
    # How many photons are in the pulse    
    Nmin = 0;Nmax = 4
    num = Nmax-Nmin+1 
    N = np.logspace(Nmin,Nmax,num)
    N = [1,2,4,8,10,20,40,80,100,200,400,800,1000,2000,4000,8000,10000,20000,40000,80000,100000]
    # What is the beam radius (Gaussian beam), (x,y)
    w1=1;w2=w1/2 #mm
    # SiPM/SPAD parameters
    PDE = [0.25]
    PDEs = len(PDE)
    pitch = 50E-3#mm
    # SiPM width/height
    dx = 5.95
    dy = 5.85
    # What simulation is running?
    #SIM = 'gauss'
    SIM = 'ellipse'
    #SIM = 'slit'
    #SIM = 'uni'
    #SIM = 'linear'
    
    # SPAD array
    xi=round(dx/pitch)
    yi=round(dy/pitch)
    # Number of SPADs
    Ncells=int(xi*yi)
      
    # Make dictionaries
    plots = {}
    data = {}
    # Make directories
    if not os.path.isdir('plots/%s'%SIM):
        os.mkdir('plots/%s'%SIM)
    
    # Main function
    for p in range(PDEs):
        for n in N:
            print('Throwing %i photons at a PDE of %.2f %%'%(n,PDE[p]*100))
            main(PDE[p],p,int(n))
 
    # End the timer
    end=time.time()

    # Print the time elapsed
    print('Simulation complete - time elapsed: %.2f hours'%((end-start)/3600))
































