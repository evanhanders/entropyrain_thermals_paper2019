import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Serif'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'DejaVu Serif'
matplotlib.rcParams['mathtext.it'] = 'DejaVu Serif:italic'
matplotlib.rcParams['mathtext.bf'] = 'DejaVu Serif:bold'
matplotlib.rcParams.update({'font.size': 9})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.interpolate import interp1d
import h5py

def theory_C(B, Gamma, V0, beta, chi, grad_T_ad=-1):
    """ The constant in stratified thermals theory """
    return np.pi**(3./2) * Gamma * grad_T_ad * chi / (V0 * B * beta**(1./2))

def theory_T(t, B, M0, I0, Gamma, V0, T0, beta, chi, grad_T_ad=-1, m_ad=1.5):
    """ The temperature profile (vs time) from stratified thermals theory """
    C0 = theory_C(B, Gamma, V0, beta, chi, grad_T_ad=grad_T_ad)
    tau = (B * t + I0)/Gamma
    tau0 = I0/Gamma
    Tpow = 1 - (m_ad/2)
    tau_int  = np.sqrt(tau )*(Gamma - (M0 - I0)/tau)
    tau_int0 = np.sqrt(tau0)*(Gamma - (M0 - I0)/tau0)
    return (2*Tpow*C0*(tau_int-tau_int0) + T0**(Tpow))**(1./Tpow)

def theory_dT_dt(t, B, M0, I0, Gamma, V0, T0, beta, chi, grad_T_ad=-1, m_ad=1.5):
    """ The temperature derivative (vs time) from stratified thermals theory """
    C0 = theory_C(B, Gamma, V0, beta, chi, grad_T_ad=grad_T_ad)
    this_T = theory_T(t, B, M0, I0, Gamma, V0, T0, beta, chi, grad_T_ad=grad_T_ad, m_ad=m_ad)
    tau = (B * t + I0)/Gamma
    dtau_dt = B/Gamma
    return C0*dtau_dt*this_T**(m_ad/2)*(Gamma/np.sqrt(tau) + (M0 - I0)/np.sqrt(tau**3))


def theory_r(t, B, I0, Gamma, beta, rho_f=None):
    """ The temperature profile (vs time) from stratified thermals theory """
    r = np.sqrt(beta*(B*t + I0)/np.pi/Gamma)
    if rho_f is None:
        return r
    return r/np.sqrt(rho_f(t))

height, width = 2.5, 3.25 

gs = gridspec.GridSpec(1000, 1000)
fig = plt.figure(figsize=(width, height))
ax  = plt.subplot(gs.new_subplotspec((0, 0), 900, 1000))


times = np.linspace(0, 100, 2000)
bouss_r = theory_r(times, -0.5, -0.32, -2, 1)

nrho_001 = (-0.552, -1e-6, -5e-6, -2.12, 7.04, 1.0, 1.00, 0.5)
nrho_001_grad_T = -(np.exp(0.01/1.5) - 1)/20
nrho_001_T = theory_T(times, *nrho_001, grad_T_ad=nrho_001_grad_T)
print(nrho_001_T)
nrho_001_d = -(nrho_001_T  - 1) / nrho_001_grad_T
nrho_001_rho = nrho_001_T**1.5
nrho_001_r = theory_r(times, nrho_001[0], -1/2, nrho_001[3], nrho_001[-2])
print(nrho_001_d)


nrho_05 = (-0.552, 0.0641, -0.32, -2.12, 7.04, 1.01, 1.00, 0.5)
nrho_05_grad_T = -(np.exp(0.5/1.5) - 1)/20
nrho_05_T = theory_T(times, *nrho_05, grad_T_ad=nrho_05_grad_T)
nrho_05_d = -(nrho_05_T  - 1) / nrho_05_grad_T
nrho_05_rho = nrho_05_T**1.5
nrho_05_r = theory_r(times, nrho_05[0], -1/2, nrho_05[3], nrho_05[-2], rho_f=interp1d(times,nrho_05_rho))

nrho_3 = (-0.879, 0.0404, -0.667, -1.73, 8.35, 1.04, 0.851, 0.5)
nrho_3_grad_T = -(np.exp(3/1.5) - 1)/20
nrho_3_T = theory_T(times, *nrho_3, grad_T_ad=nrho_3_grad_T)
nrho_3_d = -(nrho_3_T  - 1) / nrho_3_grad_T
nrho_3_rho = nrho_3_T**1.5
nrho_3_r = theory_r(times, nrho_3[0], -1/2, nrho_3[3], nrho_3[-2], rho_f=interp1d(times,nrho_3_rho))

nrho_3_r_no_buoy = theory_r(times,   0, -1/2, nrho_3[3], nrho_3[-2], rho_f=interp1d(times,nrho_3_rho))
nrho_05_r_no_buoy = theory_r(times,  0, -1/2, nrho_05[3], nrho_05[-2], rho_f=interp1d(times,nrho_05_rho))
nrho_001_r_no_buoy = theory_r(times, 0, -1/2, nrho_001[3], nrho_001[-2], rho_f=interp1d(times,nrho_001_rho))


ind_001 = nrho_001_d[nrho_001_d > 2][0] == nrho_001_d
ind_3 = nrho_3_d[nrho_3_d > 2][0] == nrho_3_d
plt.plot(nrho_001_d, (nrho_001_d/nrho_001_d[ind_001])**2, c='black', lw=0.5)
plt.plot(nrho_3_d, (nrho_3_r/nrho_3_r[ind_3])**2, c='k', lw=1.5)
plt.plot(nrho_3_d, (nrho_3_r_no_buoy/nrho_3_r_no_buoy[ind_3])**2, c='black', lw=0.5)
plt.yscale('log')
plt.ylabel(r'$f = (r/r_0)^2$')
plt.xlabel('Depth')
plt.xlim(2, 20)
plt.ylim(5e-2, 1e2)


ax.text(14, 52, r'$f \propto d^2\,\mathrm{(LJ19)}$', rotation=12)
ax.text(11.5, 1.2, r'$\mathrm{simulation}\,(n_\rho=3)$', rotation=-2)
ax.text(14, 2.2e-1, r'$f \propto \rho^{-1}\,\mathrm{(B16)}$', rotation=-7)

fig.savefig('overview_fig.png', dpi=300, bbox_inches='tight')
fig.savefig('overview_fig.pdf', dpi=600, bbox_inches='tight')
