import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

Lz = 20
n_rho       = np.logspace(-2, 1, 100) #float(args['--n_rho'])

################
# Set up atmosphere info
################
gamma     = 5./3                         #adiabatic index
m_ad      = 1/(gamma-1)
Cp        = gamma*m_ad
Cv        = Cp/gamma
grad_T_ad = -(np.exp(n_rho/m_ad) - 1)/Lz #adiabatic temperature gradient
g         = (1 + m_ad) #* -grad_T_ad     #gravity

Lr        = Lz
radius    = 0.5       #Thermal radius, by definition, in nondimensionalization
delta_r   = radius/5  #Thermal smoothing width

#(r0, z0) is the midpoint of the (spherical) thermal
r0        = 0
z0        = Lz - 3*radius

# Adjust the Re at the top of the domain if in a kappa_mu formulation so that the input Re 
# is the freefall Re at the height of the initial thermal
rho_therm = (1 + grad_T_ad*(z0 - Lz))**m_ad
H_rho_therm = -(1 + grad_T_ad*(z0-Lz))/(grad_T_ad*m_ad)
f = interp1d(n_rho, 1/H_rho_therm)

plt.loglog(n_rho, 1/H_rho_therm, lw=1)
for n in [0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 7]:
    plt.axhline(f(n), c='k', ls='--', lw=0.5)
    plt.axvline(n, c='k', ls='--',    lw=0.5)
    plt.scatter(n, f(n), c='k')
    plt.text(n*1.1, f(n)/1.5, n)

inv_f = interp1d(1/H_rho_therm, n_rho)
plt.scatter(inv_f(0.31), 0.31, 100,  c='red', marker='*')

plt.xlabel(r'$n_\rho$')
plt.ylabel(r'$L_{th}/H_\rho$')
plt.savefig('r_div_H_vs_nrho.png', dpi=200)
