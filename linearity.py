import numpy as np
from scipy import special
import matplotlib.pyplot as plt

def Nfired_func(n,reset=100E-9,pde=0.3):
    pde = 0.3
    a = (50E-6)**2*reset
    s = 1E-3/4
    A = (2*np.pi*s**2/a)
    return A*(np.euler_gamma+np.log(pde*n/A)-special.expi(-pde*n/A))

I_PD = np.array([80000,10000,960,160,21,3.2,0.35,0.045,0.007])*1E-9
I_SiPM = np.array([2500,2500,1100,600,300,210,185,176,166])*1E-9

R = 0.152
U = 3.061*1.601E-19
G = 115E-15*4
ECF = 1.1

rate_ph = I_PD/(R*U)
rate_pwr = I_PD/R*1E3
rate_eff = Nfired_func(rate_ph,reset=50E-9,pde=0.3)
rate_meas = I_SiPM/G/ECF

# Open simulation_results
mc_ph, mc_eff = np.genfromtxt('simulation_results.csv',delimiter=',',comments='#',names=True,skip_header=1,unpack=True)
print(mc_ph,mc_eff)

figure, axis = plt.subplots(2,1)
#axis2 = axis.twiny()
axis[0].plot(rate_ph,rate_eff,'o',ls='-',label='Model')
axis[0].plot(rate_ph,rate_meas,'s',ls='-',label='Measurement')
axis[0].plot(mc_ph,mc_eff,'s',ls='-',label='MC')
#axis2.plot(rate_pwr,rate_eff,'o',ls='-',label='Power')
#axis2.set_xlabel('Power [mW]')
axis[0].set_ylabel('Rate of avalanches [cps]')

axis[1].plot(rate_ph,rate_eff/rate_ph*100,'o',ls='-',label='Model')
axis[1].plot(rate_ph,rate_meas/rate_ph*100,'s',ls='-',label='Measurement')
axis[1].plot(mc_ph,mc_eff/mc_ph*100,'s',ls='-',label='MC')
#axis2.plot(rate_pwr,rate_eff,'o',ls='-',label='Power')
#axis2.set_xlabel('Power [mW]')
axis[1].set_ylabel('PDE [%]')

for ax in axis:
    ax.set_xlabel('Rate of photons [cps]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()

plt.show()
