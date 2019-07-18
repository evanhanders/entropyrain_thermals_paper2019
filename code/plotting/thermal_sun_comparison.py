import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Serif'
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
matplotlib.rcParams['mathtext.rm'] = 'DejaVu Serif'
matplotlib.rcParams['mathtext.it'] = 'DejaVu Serif:italic'
matplotlib.rcParams['mathtext.bf'] = 'DejaVu Serif:bold'
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d


cm2Mm = 1e-8 #Mm / cm
Rsun2cm = 6.957e10 #cm / Rsun
Rsun2Mm = 6.957e2 #Mm / Rsun

gs = gridspec.GridSpec(1000, 1000)
fig = plt.figure(figsize=(3.25, 7))




def theory_C(B0, Gamma0, f, chi, beta, grad_T_ad=-1):
    """ The constant in stratified thermals theory """
    return (grad_T_ad * Gamma0 * np.pi**(3./2) / f / (1 + beta)) * np.sqrt(Gamma0/(B0 * chi))

def theory_r(t, B0, Gamma0, chi, t_off, rho_f=None):
    """ The temperature profile (vs time) from stratified thermals theory """
    r = np.sqrt(chi*B0*(t + t_off)/np.pi/Gamma0)
    if rho_f is None:
        return r
    return r/np.sqrt(rho_f(t))

def theory_T(t, B0, Gamma0, f, chi, beta, t_off, T0, grad_T_ad=-1, m_ad=1.5):
    """ The temperature profile (vs time) from stratified thermals theory """
    C0 = theory_C(B0, Gamma0, f, chi, beta, grad_T_ad=grad_T_ad)
    Tpow = 1 - (m_ad/2)
    return (2*Tpow*C0*np.sqrt(t + t_off) + T0**(Tpow))**(1./Tpow)

def theory_dT_dt(t, B0, Gamma0, f, chi, beta, t_off, T0, grad_T_ad=-1, m_ad=1.5):
    """ The temperature derivative (vs time) from stratified thermals theory """
    C0 = theory_C(B0, Gamma0, f, chi, beta, grad_T_ad=grad_T_ad)
    this_T = theory_T(t, B0, Gamma0, f, chi, beta, t_off, T0, grad_T_ad=grad_T_ad, m_ad=m_ad)
    return C0*this_T**(m_ad/2)/np.sqrt(t + t_off)

def read_solar_profiles():
    mesa_data = np.genfromtxt('sun_4.5gyr.data', skip_header=6, usecols=(2, 4, 9, 10, 11)) #logR (in Rsun units), logRho (g/cm^3), grav (cm/s^2), Cp, csound (cm/s)
    grav = mesa_data[:,2]
    Cp = mesa_data[:,3] #erg/K?
    csound = mesa_data[:,4]*cm2Mm
    mesa_data = 10**(mesa_data[:,:-1])
    radius    = mesa_data[:,0]*Rsun2Mm
    density    = mesa_data[:,1]*(cm2Mm)**(-3)

    pm_data   = np.genfromtxt('Pm.csv', delimiter=',')
    pr_data   = np.genfromtxt('Pr.csv', delimiter=',')
    eta_data   = np.genfromtxt('eta_sun.csv', delimiter=',')

    pm_sort = np.argsort(pm_data[:,0])
    pr_sort = np.argsort(pr_data[:,0])
    eta_sort = np.argsort(eta_data[:,0])

    r_pm, pm = pm_data[pm_sort,0], pm_data[pm_sort,1]
    r_pr, pr = pr_data[pr_sort,0], pr_data[pr_sort,1]
    r_eta, eta = eta_data[eta_sort,0], eta_data[eta_sort,1]*1e-7

    pm_f = interp1d(r_pm, pm, bounds_error=False, fill_value='extrapolate')
    pr_f = interp1d(r_pr, pr, bounds_error=False, fill_value='extrapolate')
    eta_f = interp1d(r_eta, eta, bounds_error=False, fill_value='extrapolate')

    radii = np.linspace(0.68, 0.995, 1000)
    chi  = pm_f(radii) * eta_f(radii) / pr_f(radii) * (cm2Mm)**2
    nu  = pm_f(radii) * eta_f(radii) * (cm2Mm)**2
    radii *= Rsun2Mm

    rho2r = interp1d(density, radius, bounds_error=False, fill_value='extrapolate')
    r2chi = interp1d(radii, chi, bounds_error=False, fill_value='extrapolate')
    r2nu  = interp1d(radii, nu, bounds_error=False, fill_value='extrapolate')


    r2cs = interp1d(radius, csound, bounds_error=False, fill_value='extrapolate')
    r2gDivCp = interp1d(radius, grav/Cp, bounds_error=False, fill_value='extrapolate')
    r2Cp = interp1d(radius, Cp, bounds_error=False, fill_value='extrapolate')

    return rho2r, r2chi, r2nu, r2cs, r2gDivCp, r2Cp

rho2r, r2chi, r2nu, r2cs, r2gDivCp, r2Cp = read_solar_profiles()



# Values from AL08 (Avrett & Loeser 2008) are taken to be the values at h = 35 km, the height at which T = 6000 K
# rho_mean is int ( rho ) dz / L from h = 35-L to h = 35
# Value for T1 is given in BBR02 (Borrero & Bellot Rubio 2002), from Valentin


#Thermodynamics
T0          = 6000       #K
T1          = -500       #K BBR02
rho_top     = 1.74e-7 * (cm2Mm)**-3 #g/Mm^3, AL08
rho_mean    = 1.169*rho_top         #g/Mm^3, AL08
gamma       = 5./3
S1_over_cp  = (1./gamma) * np.log(1 + T1/T0)
epsilon     = np.abs(S1_over_cp)

#Other parameters
L           = 0.1        #Mm
g           = 2.74e-4    #Mm/s^2
#Need Cp. Sadness.
cs          = 9.5e-3     #Mm/s, AL08
u_th        = np.sqrt(epsilon)*cs


R_approx    = cs**2/T0


B       = (4.*np.pi/3) * g * (L/2)**3 * S1_over_cp * rho_mean
nondim  = rho_top * L**2 * u_th**2 * np.abs(S1_over_cp) #g * Mm / s^2

print('dimensional B: {:.4e}, nondim B: {:.4e}'.format(B, B/nondim))

#nrho, z_{th,0}, t_off, B, Gamma, m, chi, k
data = np.array([
(0.1, 24.6, 0.144, -0.548, -2.17, 8.05, 1.04 , 0.732),
(0.5, 24.2, 0.695, -0.569, -2.12, 8.34, 0.977, 0.715),
(1,   23.7, 1.11,  -0.602, -2.05, 8.65, 0.915, 0.703),
(2,   22.3, 1.27,  -0.713, -1.89, 9.23, 0.842, 0.682),
(3,   21.2, 1.01,  -0.947, -1.73, 9.81, 0.807, 0.654),
(4,   20.5, 0.615, -1.47,  -1.59, 10.2, 0.794, 0.642),
(5,   20.0, 0.425, -2.70,  -1.49, 10.7, 0.781, 0.609),
(6,   19.8, 0.041, -5.73,  -1.43, 10.8, 0.787, 0.616)
])

times = np.logspace(-1, 6, 2000)
m_ad = 1/(gamma-1)
rho_outs = []
r_outs   = []
T_outs   = []
w_outs   = []
fits_outs = []

B_nondim = B/nondim

for Buoy in (B_nondim/2, 2*B_nondim):
    dataset = []
    for i in range(data.shape[1]):
        if i == 3:
            dataset.append(Buoy)
            continue
        else:
            func = interp1d(data[:,3], data[:,i], fill_value='extrapolate', bounds_error=False)
            dataset.append(func(Buoy))

    n_rho, z0, t_off, B0, Gamma, m, chi, k = dataset
    if t_off < 0:
        t_off = 0
    grad_T_ad = -(np.exp(n_rho/m_ad)-1)/20
    Tatm = 1 + grad_T_ad*(z0-20)

    Temps = theory_T(times, B0, Gamma, m, chi, k, t_off, Tatm, grad_T_ad=grad_T_ad, m_ad=m_ad)
    Rhos  = Temps**(m_ad)
    ws    = theory_dT_dt(times, B0, Gamma, m, chi, k, t_off, Tatm, grad_T_ad=grad_T_ad, m_ad=m_ad)/grad_T_ad

    rho_f = interp1d(times, Rhos)
    rs    = theory_r(times, B0, Gamma, chi, t_off, rho_f=rho_f)

    rho_outs.append(Rhos)
    T_outs.append(Temps)
    w_outs.append(ws)
    r_outs.append(rs)
    fits_outs.append(dataset)


fig = plt.figure(figsize=(3.25, 6))
#######################################################AX1
ax = plt.subplot(gs.new_subplotspec((0,  0), 200, 1000))

#Plot radius grey band
x_values = rho_outs[0]*(rho_top/rho_outs[0][0])
upper_y_values = r_outs[0]*(L/2/r_outs[0][0])
lower_y_values = interp1d(rho_outs[1]*(rho_top/rho_outs[1][0]), r_outs[1]*(L/2/r_outs[1][0]))(x_values)
plt.fill_between(rho2r(x_values)/(Rsun2Mm), lower_y_values, upper_y_values, color='black', alpha=0.4, rasterized=True)

#plot actual radius measurements
solar_radii = []
solar_therm_radii = []
solar_rhos = []
for i in range(2):
    solar_rhos.append(rho_outs[i]*(rho_top/rho_outs[i][0]))
    solar_radii.append(rho2r(solar_rhos[i]))
    solar_therm_radii.append(r_outs[i]*(L/2/r_outs[i][0]))
    plt.plot(solar_radii[i]/(Rsun2Mm), solar_therm_radii[i], c='k', lw=0.5*(i+1))

plt.plot(solar_radii[0]/(Rsun2Mm), (L/2)/(rho_outs[0]/rho_outs[0][0])**(1./2), c='k', lw=2)
ax.text(0.69, 1e-4, r'$r \propto \rho^{-1/2}$', rotation=3)
ax.text(0.69, 8e-3, 'thermals', rotation=2)

#Make plot pretty
plt.xlim(0.68, 0.995)
plt.yscale('log')
plt.ylim(1e-5, 1e-1)
plt.ylabel(r'$r_{\mathrm{th}}$ (Mm)')

#######################################################AX2
ax = plt.subplot(gs.new_subplotspec((200,  0), 200, 1000))
#Plot radius grey band
x_values = solar_radii[0] 
upper_y_values = w_outs[0]*(u_th/w_outs[0][0])
lower_y_values = interp1d(solar_radii[1], w_outs[1]*(u_th/w_outs[1][0]))(x_values)
plt.fill_between(x_values/(Rsun2Mm), lower_y_values, upper_y_values, color='black', alpha=0.4, rasterized=True)

#plot actual velocity measurements
solar_therm_w = []
for i in range(2):
    solar_therm_w.append(w_outs[i]*(u_th/w_outs[i][0]))
    plt.plot(solar_radii[i]/(Rsun2Mm), solar_therm_w[i], c='k', lw=0.5*(i+1))
plt.plot(solar_radii[i]/Rsun2Mm, r2cs(solar_radii[i]), c='k', lw=1, dashes=(5,1,2,1))
ax.text(0.69, 3.5e-1, r'$c_s$', rotation=-3)

#Make plot pretty
plt.xlim(0.68, 0.995)
plt.ylim(1e-3, 7e-1)
plt.yscale('log')
plt.ylabel(r'$w_{\mathrm{th}}$ (Mm/s)')


#######################################################AX3
ax = plt.subplot(gs.new_subplotspec((400,  0), 200, 1000))

tau_ratios = []
for i in range(2):
    chi    = r2chi(solar_radii[i])
    nu     = r2nu(solar_radii[i])

    tau_kappa = r_outs[i]*(L/2/r_outs[i][0])**2/chi
    tau_ff    = -r_outs[i]/(w_outs[i]*u_th)
    tau_ff_200    = -200/(w_outs[i]*u_th)
#    tau_ff    = -1/(w_outs[i]*u_th)
    tau_ratios.append(tau_kappa/tau_ff)

    good = solar_radii[i]/Rsun2Mm > 0.7
    fractional_diffusion = tau_kappa[:-1]**(-1) * np.diff(solar_radii[i]) / (w_outs[i]*u_th)[:-1]
    print('fractional diffusion: {:.4e}'.format(np.sum(fractional_diffusion[good[:-1]])))

    viscous_heating = (nu*(w_outs[i]*u_th)**2/(r_outs[i]*(L/2/r_outs[i][0]))**2 / T_outs[i]**(2.5))[:-1] * np.diff(solar_radii[i]) / (w_outs[i]*u_th)[:-1]
    print('VH / (S1/cp): {:.4e}'.format(np.sum(viscous_heating[good[:-1]]/(R_approx*2.5)/S1_over_cp)))


#Plot grey band
x_values = solar_radii[0] 
upper_y_values = tau_ratios[0]
lower_y_values = interp1d(solar_radii[1], tau_ratios[1])(x_values)
plt.fill_between(x_values/Rsun2Mm, lower_y_values, upper_y_values, color='black', alpha=0.4, rasterized=True)
for i in range(2):
    plt.plot(solar_radii[i]/(Rsun2Mm), tau_ratios[i] , c='k', lw=(0.5)*(i+1))

plt.xlim(0.68, 0.995)
plt.yscale('log')
plt.ylabel(r'$\tau_\kappa/\tau_{\mathrm{ff}}$')
plt.xlabel(r'radius $(R_{\odot})$')
#plt.legend(loc='best', fontsize=9, frameon=False)
plt.ylim(3e5, 3e8)


#######################################################AX4
ax = plt.subplot(gs.new_subplotspec((600,  0), 200, 1000))
L_enth_fracs = []
for i in range(2):
    solar_T = T_outs[i]*(T0/T_outs[i][0])
    V = fits_outs[i][-3]*(solar_therm_radii[i]/cm2Mm)**3
    rho = solar_rhos[i]*cm2Mm**3
    w = -solar_therm_w[i]/cm2Mm
    cross_area = np.pi*solar_therm_radii[i]**2/cm2Mm**2
    g_over_Cp = r2gDivCp(solar_radii[i])
    B = (fits_outs[i][3]*nondim/cm2Mm)
    sfluc = B / (rho*V*g_over_Cp)

#    L_ke = 0.5*rho*w**3*cross_area
    L_enth = rho*solar_T*sfluc*w*cross_area
    L_enth_fracs.append(L_enth/3.84e33)

#Plot grey band
x_values = solar_radii[0] 
upper_y_values = L_enth_fracs[0]
lower_y_values = interp1d(solar_radii[1], L_enth_fracs[1])(x_values)
plt.fill_between(x_values/Rsun2Mm, lower_y_values, upper_y_values, color='black', alpha=0.4, rasterized=True)
for i in range(2):
    plt.plot(solar_radii[i]/(Rsun2Mm), L_enth_fracs[i], c='k', lw=0.5*(i+1))


plt.xlim(0.68, 0.995)
plt.ylim(1e-10, 1e-4)
plt.yscale('log')
plt.ylabel(r'$L_{\mathrm{enth, th}}/L_{\odot}$')


#######################################################AX4
ax = plt.subplot(gs.new_subplotspec((800,  0), 200, 1000))
f_thermals = []
for i in range(2):
    cross_area = np.pi*solar_therm_radii[i]**2
    surf_area  = 4*np.pi*solar_radii[i]**2
    N_thermals = L_enth_fracs[i]**(-1)
    A_thermals = N_thermals*cross_area
    f_thermals.append(A_thermals/surf_area)

#Plot grey band
x_values = solar_radii[0] 
upper_y_values = f_thermals[0]
lower_y_values = interp1d(solar_radii[1], f_thermals[1])(x_values)
plt.fill_between(x_values/Rsun2Mm, lower_y_values, upper_y_values, color='black', alpha=0.4, rasterized=True)
for i in range(2):
    plt.plot(solar_radii[i]/Rsun2Mm, f_thermals[i], c='k', lw=0.5*(i+1))



plt.xlim(0.68, 0.995)
plt.ylim(1e-7, 1e0)
plt.yscale('log')
plt.xlabel(r'radius $(R_{\odot})$')
plt.ylabel(r'$f_{\mathrm{th}}$')



plt.savefig('thermal_sun_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('thermal_sun_comparison.pdf', dpi=600, bbox_inches='tight')

