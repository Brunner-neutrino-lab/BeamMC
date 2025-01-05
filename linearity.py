import numpy as np
from scipy import special
import matplotlib.pyplot as plt

def Nfired_func(n,reset=100E-9):
    k = 0.3
    a = (50E-6)**2*reset
    s = 1E-3/4
    A = (2*np.pi*s**2/a)
    return A*(np.euler_gamma+np.log(k*n/A)-special.expi(-k*n/A))

I_PD = np.array([80000,10000,960,160,21,3.2,0.35,0.045,0.007])*1E-9
I_SiPM = np.array([2500,2500,1100,600,300,210,185,176,166])*1E-9

R = 0.152
U = 3.061*1.601E-19
G = 115E-15*4
ECF = 1.1

rate_ph = I_PD/(R*U)
rate_pwr = I_PD/R*1E3
rate_eff = Nfired_func(rate_ph,reset=50E-9)
rate_meas = I_SiPM/G/ECF

# Open simulation_results
mc_ph, mc_eff = np.genfromtxt('simulation_results.csv',delimiter=',',comments='#',names=True,skip_header=1,unpack=True)
print(mc_ph,mc_eff)

figure, axis = plt.subplots(1,1)
#axis2 = axis.twiny()
axis.plot(rate_ph,rate_eff,'o',ls='-',label='Model')
axis.plot(rate_ph,rate_meas,'s',ls='-',label='Measurement')
axis.plot(mc_ph,mc_eff,'s',ls='-',label='MC')
#axis2.plot(rate_pwr,rate_eff,'o',ls='-',label='Power')
axis.set_xlabel('Rate of photons [cps]')
#axis2.set_xlabel('Power [mW]')
axis.set_ylabel('Rate of avalanches [cps]')
axis.set_xscale('log')
#axis2.set_xscale('log')
axis.set_yscale('log')
axis.legend()
plt.show()
