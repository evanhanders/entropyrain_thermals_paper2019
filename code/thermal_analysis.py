"""
Analyzes a 2D, cylindrically symmetric thermal's properties 

Usage:
    thermal_analysis.py --root_dir=<dir> [options]

Options:
    --root_dir=<root_dir>               Directory pointing to 'slices', etc
    --plot                              If flagged, plot colormesh during analysis
    --no_analyze                        If flagged, do not do full analysis
    --n_files=<n>                           Number of files [default: 10000]
    --get_contour                       If flagged, do full contour solve
    --iterative                         If flagged, use results of previous solve to inform new solve
"""
import os
import glob
import logging
logger = logging.getLogger(__name__)

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.optimize import brentq
import scipy.optimize as scop

from dedalus import public as de

from logic.post_processing import * #pull in most of the functions used here

FILETYPE='.png'

def post_process(root_dir, plot=False, get_contour=True, analyze=True, out_dir='thermal_analysis', n_files=int(1e6), iterative=False):
    """
    Inputs:
    -------
    root_dir    : string
        The root directory of the dedalus run, containing folders like slices/
    out_dir     : string
        The directory inside of root_dir in which output is saved
    """
    #### Step 1.
    #Set up general run info
    post = ThermalPostProcessor(root_dir, out_dir)

    #### Step 2.
    #If flagged, solve thermal tracking algorithm to get contour
    if get_contour:
        post.measure_cb()
        post.fit_w_cb(iterative=iterative)
        post.calculate_contour(iterative=iterative)

    #### Step 3.
    #If flagged, plot colormeshes
    if plot:
        post.plot_colormeshes()
    
    #### Step 4.
    #If the thermal contour is found, this section calculates quantities inside (or outside) of the thermal from sim outputs
    if analyze:
        post.measure_values_in_thermal()

    #### Step 5.
    #Read in post-processed data and make plots

    #Read in integrated quantities
    output_file  = h5py.File('{:s}/post_analysis.h5'.format(post.full_out_dir), 'r')
    times        = output_file['times'].value
    int_circ     = output_file['int_circ'].value
    int_circ_a   = output_file['int_circ_above'].value
    int_impulse  = output_file['int_impulse'].value
    int_rho_s1_a = output_file['int_rho_s1_above'].value
    int_rho_s1   = output_file['int_rho_s1'].value
    int_mom      = output_file['int_mom'].value
    volumes      = output_file['int_vol'].value
    area         = output_file['int_area'].value
    radius       = output_file['radius'].value
    output_file.close()

    #Read in fit quantities
    fit_file = h5py.File('{:s}/fit_file.h5'.format(post.full_out_dir), 'r')
    fit_B     = fit_file['fit_B'].value 
    fit_M0    = fit_file['fit_M0'].value  
    fit_I0    = fit_file['fit_I0'].value  
    fit_beta  = fit_file['fit_beta'].value    
    fit_chi   = fit_file['fit_chi'].value    
    fit_Gamma = fit_file['fit_Gamma'].value     
    fit_V0    = fit_file['fit_V0'].value  
    fit_T0    = fit_file['fit_T0'].value  

    w_cb         = fit_file['w_cb'].value
    cb_T_fit     = fit_file['cb_T_fit'].value
    T0 = vortex_T     = fit_file['vortex_T'].value
    vortex_w     = fit_file['vortex_w'].value
    rho0 = vortex_rho   = fit_file['vortex_rho'].value
    therm_radius = fit_file['vortex_radius'].value
    height       = fit_file['vortex_height'].value
    fit_file.close()
    height = z_measured = (T0 - 1)/post.grad_T_ad + post.Lz

    theory_impulse  = linear_fit(post.times, fit_B*fit_beta,   fit_I0*fit_beta)
    theory_momentum = linear_fit(post.times, fit_B*fit_beta*fit_chi, fit_M0*fit_beta*fit_chi)
    theory_radius   = np.sqrt(theory_impulse/(np.pi*vortex_rho*fit_Gamma))
   
    #Read in contours
    contour_file = h5py.File('{:s}/contour_file.h5'.format(post.full_out_dir), 'r')
    z        = contour_file['z'].value
    contours = contour_file['contours'].value
    contour_file.close()

    thermal_found = np.zeros(contours.shape[0], dtype =bool) 
    for i in range(contours.shape[0]):
        if not (contours[i,-1] != 0)*(contours[i,-2]==0):
            thermal_found[i] = True
    good = thermal_found

    #Find Rz based on contour 
    Rz = np.zeros_like(times)
    z_bots = np.zeros_like(times)
    for i in range(len(times)):
        max_c = contours[i].max()
        if len(contours[i][contours[i] > 0.05*max_c]) < 2:
            Rz[i] = 0
            continue
        z_bot = z[contours[i] > 0.05*max_c][0]
        if np.sum(contours[i][z > z_bot] < 0.05*max_c) == 0:
            Rz[i] = 0
            continue
        z_top = z[z>z_bot][contours[i][z>z_bot] < 0.05*max_c][0]
        Rz[i] = (z_top - z_bot)/2
        z_bots[i] = z_bot

    #cb fit
    depth = post.Lz - height
    fit_t = (height < 0.65*post.Lz)*(height > 0.1*post.Lz)
    fit_t[0] = False
    found_therm = False
    therm_done  = False
    for i in range(len(times)):
        if therm_done:
            fit_t[i] = False
            good[i] = False
        if fit_t[i] and not found_therm:
            found_therm = True
        if found_therm and not fit_t[i] and not therm_done:
            therm_done=True
    good[0] = False

    scale_t = times

    (dB, B0), pcov = scop.curve_fit(linear_fit, scale_t[fit_t], int_rho_s1[fit_t])
    best_fit_B = B0 + dB*scale_t[fit_t][-1]/2
    (dGamma, Gamma0), pcov = scop.curve_fit(linear_fit, scale_t[fit_t], int_circ[fit_t])
    best_fit_Gamma = Gamma0 + dGamma*scale_t[fit_t][-1]/2
    best_fit_beta = fit_beta * (fit_B/best_fit_B) * (best_fit_Gamma/fit_Gamma)
    p_B = int_mom/best_fit_B/(best_fit_beta)/fit_chi
    (chi_adjust, M0_div_B), pcov = scop.curve_fit(linear_fit, scale_t[fit_t], p_B[fit_t])
    best_fit_chi = fit_chi*chi_adjust 

#    if fit_M0 > 0:
#        M0_min, M0_max = -fit_M0, 2*fit_M0
#    else:
#        M0_min, M0_max = 2*fit_M0, -fit_M0
#    if fit_I0 > 0:
#        I0_min, I0_max = -fit_I0, 2*fit_I0
#    else:
#        I0_min, I0_max = 2*fit_I0, -fit_I0
#    bounds = ((0.7*T0[0], 0.7*fit_V0, M0_min, I0_min), 
#              (1.3*T0[0], 1.3*fit_V0, M0_max, I0_max ))
#
    bounds = ((0*T0[0], 0.7*fit_V0), 
              (2*T0[0], 1.3*fit_V0 ))
    p0 = (T0[0], fit_V0)
    this_T_theory      = lambda times, T, V:  theory_T_no_I0(    times, best_fit_B, best_fit_Gamma, V, T, best_fit_beta, best_fit_chi, grad_T_ad=post.grad_T_ad, m_ad=post.m_ad)
#
    this_fit = good*post.get_good_times(height, L_max=0.5, L_min=0.1)
    best_fit_M0 = best_fit_I0 = 0
    (best_fit_T0, best_fit_V0), pcov = scop.curve_fit(this_T_theory, scale_t[this_fit], T0[this_fit], bounds=bounds, p0=p0, maxfev=1e4)
#    try:
#        (best_fit_T0, best_fit_V0, best_fit_M0, best_fit_I0), pcov = scop.curve_fit(this_T_theory, scale_t[this_fit], T0[this_fit], bounds=bounds, p0=p0, maxfev=1e4)
#    except:
#        bounds = ((0.7*T0[0], 0.5*fit_V0, M0_min, I0_min), 
#                  (1.3*T0[0], 1.5*fit_V0, M0_min/10, I0_min/10 ))
#        (best_fit_T0, best_fit_V0, best_fit_M0, best_fit_I0), pcov = scop.curve_fit(this_T_theory, scale_t[this_fit], T0[this_fit], bounds=bounds, p0=p0, maxfev=1e4)
#        best_fit_T0, best_fit_V0, best_fit_M0, best_fit_I0 = fit_T0, fit_V0, fit_M0, fit_I0

    best_theory_impulse  = linear_fit(post.times, best_fit_B*best_fit_beta,   fit_I0*best_fit_beta)
    best_theory_momentum = linear_fit(post.times, best_fit_B*best_fit_beta*best_fit_chi, fit_M0*best_fit_beta*best_fit_chi)
    best_theory_radius   = np.sqrt(best_theory_impulse/(np.pi*rho0*best_fit_Gamma))
    best_theory_T        = theory_T_no_I0(scale_t, best_fit_B, best_fit_Gamma, best_fit_V0, best_fit_T0, best_fit_beta, best_fit_chi, grad_T_ad=post.grad_T_ad, m_ad=post.m_ad)
    best_theory_w        = theory_dT_dt_no_I0(scale_t, best_fit_B, best_fit_Gamma, best_fit_V0, best_fit_T0, best_fit_beta, best_fit_chi, grad_T_ad=post.grad_T_ad, m_ad=post.m_ad)/post.grad_T_ad

    #Eqn1, B = \int \rho S_1 dV = const
    logger.info('plotting eqn1, integrated entropy')
    fig, axs, B_fit, (a, B0) = plot_and_fit_trace(scale_t[good], int_rho_s1[good], fit_ind=fit_t[good], fit_func=linear_fit, 
                       labels=['t',r'$\int (\rho S_1) dV$'],
                       fit_str_format='{:.2g} $t$ + {:.2g}')
    axs[0].axhline(fit_B, c='green', dashes=(2,1))
    axs[0].axhline(best_fit_B, c='mediumorchid', dashes=(2,1))
    logger.info('entropy frac change: {}'.format(a*scale_t[fit_t][-1]/B0))
    fig.savefig('{:s}/tot_entropy_v_time{:s}'.format(post.full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
    plt.close(fig)


    #Eqn4, volume ~ R^3, or r^3
    logger.info('plotting eqn4, V/r^3 = V_0')
    V0 = volumes/therm_radius**3
    fig, axs, V0_fit, (dV0_dt, V0) = plot_and_fit_trace(scale_t[good*np.isfinite(V0)], V0[good*np.isfinite(V0)], fit_ind=fit_t[good*np.isfinite(V0)], fit_func=linear_fit, 
                       labels=['t',r'$V_0 = V/r^3$'],
                       fit_str_format='{:.2g} $t$ + {:.2g}')
    axs[0].axhline(fit_V0, c='green', dashes=(2,1))
    axs[0].axhline(best_fit_V0, c='mediumorchid', dashes=(2,1))
    fig.savefig('{:s}/V0_v_time{:s}'.format(post.full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
    plt.close(fig)

    #Eqn 7, rho V w / B = beta t + M_0
    logger.info('plotting eqn7, rho V w / B = linear')
    p_B = int_mom/fit_B/fit_beta/fit_chi
    fig, axs, momentum_div_B, (beta, M0_div_B) = plot_and_fit_trace(scale_t[good*np.isfinite(p_B)], p_B[good*np.isfinite(p_B)], fit_ind=fit_t[good*np.isfinite(p_B)], fit_func=linear_fit, 
                       labels=['t',r'$\int \rho w dV / B / (\beta\chi)$'],
                       fit_str_format='{:.2g} $t$ + {:.2g}')
    axs[0].plot(post.times, theory_momentum/fit_B/(fit_beta)/fit_chi, c='green', dashes=(2,1))
    axs[0].plot(post.times, best_theory_momentum/fit_B/(fit_beta)/fit_chi, c='mediumorchid', dashes=(2,1))
    fig.savefig('{:s}/momentum_div_B_v_time{:s}'.format(post.full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
    plt.close(fig)

    #Eqn9, B = \int omega dA = const
    logger.info('plotting eqn9, circulation')
    fig, axs, Gamma_fit, (a, Gamma0) = plot_and_fit_trace(scale_t[good],int_circ[good], fit_ind=fit_t[good], fit_func=linear_fit, 
                       labels=['t',r'$\Gamma_{thermal}$'],
                       fit_str_format='{:.2g} $t$ + {:.2g}')
    axs[0].axhline(fit_Gamma, c='green', dashes=(2,1))
    axs[0].axhline(best_fit_Gamma, c='mediumorchid', dashes=(2,1))
    fig.savefig('{:s}/circulation_v_time{:s}'.format(post.full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
    plt.close(fig)

    #Eqn 8, 0.5*rho0*r^2*Gamma / B = t + Const
    logger.info('plotting eqn8, 0.5* rho r^2 Gamma / B = t + const')
    I_B = np.pi*therm_radius**2*fit_Gamma*rho0/fit_B/fit_beta
#    I_B = int_impulse/fit_B/fit_beta
    fig, axs, I_B_fit, (I0_f, I0_div_B) = plot_and_fit_trace(scale_t[good*np.isfinite(I_B)], I_B[good*np.isfinite(I_B)], fit_ind=fit_t[good*np.isfinite(I_B)], fit_func=linear_fit, 
                       labels=['t',r'$\pi\rho r^2 \Gamma / B / \beta$'],
                       fit_str_format='{:.2g} $t$ + {:.2g}')
    axs[0].plot(post.times, theory_impulse/fit_B/fit_beta, c='green', dashes=(2,1))
    axs[0].plot(post.times, best_theory_impulse/fit_B/fit_beta, c='mediumorchid', dashes=(2,1))
    fig.savefig('{:s}/impulse_div_B_v_time{:s}'.format(post.full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
    plt.close(fig)

    #Eqn 10, r = sqrt ( 2 * [Bt + I_0] / rho / Gamma )
    logger.info('plotting eqn10, radius fit')

    fig = plt.figure()
    ax = fig.add_subplot(2,1,1) 
    plt.scatter(scale_t[good], therm_radius[good], label='r', marker='+', c='k')
    plt.plot(scale_t[good], theory_radius[good], label=r'$\sqrt{{\frac{{\beta(Bt + I_0)}}{{\pi \rho \Gamma}}}}$', color='green')
    plt.plot(scale_t[good], best_theory_radius[good], label=r'$\sqrt{{\frac{{\beta(Bt + I_0)}}{{\pi \rho \Gamma}}}}$', color='mediumorchid', ls='--')
    plt.ylim(0, 1.25*np.max(therm_radius))
    plt.ylabel('r')
    plt.legend(loc='best')
    ax = fig.add_subplot(2,1,2)
    plt.plot(scale_t[good], np.abs(1 - therm_radius[good]/theory_radius[good]), c='green')
    plt.plot(scale_t[good], np.abs(1 - therm_radius[good]/best_theory_radius[good]), c='mediumorchid', ls='--')
    plt.yscale('log')
    plt.ylabel('fractional diff')
    plt.xlabel('t')
    fig.savefig('{:s}/r_Fit_v_time.png'.format(post.full_out_dir), bbox_inches='tight', dpi=200)


    #Eqn 11, 
    logger.info('plotting eqn 4 & 11, momentum fit')
    momentum_est = theory_momentum
    momentum_linear = (best_fit_beta/2)*(best_fit_B*scale_t+best_fit_M0)

    fig = plt.figure()
    ax = fig.add_subplot(2,1,1) 
    plt.plot(scale_t[good], int_mom[good], c='k', label=r'$\int \rho w dV$')
    plt.plot(scale_t[good], theory_momentum[good], c='green', ls='--', label=r'old fit')
    plt.plot(scale_t[good], best_theory_momentum[good], c='mediumorchid', label=r'new fit')
    if fit_B < 0:
        plt.ylim(int_mom[good].min()*1.25, 0)
    else:
        plt.ylim(0, int_mom[good].max()*1.25)

    plt.ylabel('momentum')
    plt.legend(loc='best')
    ax = fig.add_subplot(2,1,2)
    plt.grid(which='major')
    plt.plot(scale_t[good], np.abs(1 - best_theory_momentum[good]/int_mom[good]), c='mediumorchid', label='theory V measured')
    plt.plot(scale_t[good], np.abs(1 - theory_momentum[good]/int_mom[good]), c='orange', label='linear V measured')
    plt.ylim(1e-4, 1e0)
    plt.yscale('log')
    plt.ylabel('fractional diff')
    plt.xlabel('t')
    fig.savefig('{:s}/p_Fit_v_time.png'.format(post.full_out_dir), bbox_inches='tight', dpi=200)


    w_theory_2 = differentiate(scale_t, best_theory_T)/post.grad_T_ad#best_theory_momentum / rho0 / best_fit_V0 / best_theory_radius**3

    #Outputs: z(t), r(z), w(z)
    logger.info('plotting z(t)')
    z_theory   = ((cb_T_fit - 1)/post.grad_T_ad + post.Lz)
    new_z_theory   = ((best_theory_T - 1)/post.grad_T_ad + post.Lz)
    z_measured = ((vortex_T - 1)/post.grad_T_ad + post.Lz)

    d_theory   = post.Lz - z_theory
    new_d_theory   = post.Lz - new_z_theory
    d_measured = post.Lz - z_measured
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1) 
    plt.scatter(scale_t[good], d_measured[good], c='k', label=r'$L_z - z$ (measured)', marker='+')
    plt.plot(scale_t[good], d_theory[good], c='orange', label=r'$L_z - z$ (theory)')
    plt.plot(scale_t[good], new_d_theory[good], c='mediumorchid', label=r'$L_z - z$ (new theory)')
    plt.ylabel('depth')
    plt.legend(loc='best')
    plt.ylim(0, post.Lz)
    ax = fig.add_subplot(2,1,2)
    plt.grid(which='major')
    plt.scatter(scale_t[good], np.abs(1 - d_theory[good]/d_measured[good]), c='orange', label='theory V measured')
    plt.scatter(scale_t[good], np.abs(1 - new_d_theory[good]/d_measured[good]), c='mediumorchid', label='new theory V measured', marker='x')
    plt.ylim(1e-4, 1e0)
    plt.yscale('log')
    plt.ylabel('fractional diff')
    plt.xlabel('t')

    fig.savefig('{:s}/z_v_t.png'.format(post.full_out_dir), bbox_inches='tight', dpi=200)



    logger.info('plotting r(z)')
    r_measured = therm_radius
    r_theory   = theory_radius
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1) 
    plt.scatter(d_measured[good], r_measured[good], c='k', label=r'$r$ (measured)', marker='+')
    plt.plot(d_theory[good],   r_theory[good], c='orange', label=r'$r$ (theory)')
    plt.plot(new_d_theory[good],   best_theory_radius[good], c='mediumorchid', label=r'$r$ (new theory)')
    plt.ylim(np.min(r_measured[good])*0.75, np.max(r_measured[good])*1.25)
    plt.xlim(0, post.Lz)
    plt.ylabel('r')
    plt.legend(loc='best')
    plt.xlabel('depth')
    ax = fig.add_subplot(2,1,2)
    plt.grid(which='major')
    plt.scatter(scale_t[good], np.abs(1 - r_theory[good]/r_measured[good]), c='orange', label='theory V measured')
    plt.scatter(scale_t[good], np.abs(1 - best_theory_radius[good]/r_measured[good]), c='mediumorchid', label='new theory V measured', marker='x')
    plt.ylim(1e-4, 1e0)
    plt.yscale('log')
    plt.ylabel('fractional diff')
    plt.xlabel('t')


    fig.savefig('{:s}/r_v_z.png'.format(post.full_out_dir), bbox_inches='tight', dpi=200)

    logger.info('plotting w(z)')
    w_measured = vortex_w#differentiate(scale_t, z_measured)
    w_m_2      = np.diff(z_measured)/np.diff(scale_t)
    w_theory   = w_cb[2:-2]
    this_good  = good[2:-2]
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1) 
    plt.scatter(d_measured[2:-2][this_good], w_measured[this_good], c='k', marker='+', label=r'$w$ (measured)')
    plt.plot(d_theory[2:-2][this_good],  w_theory[this_good],   c='orange', label=r'$w$ (theory)')
    plt.plot(new_d_theory[2:-2],  best_theory_w[2:-2],   c='mediumorchid', label=r'$w$ (new theory)')
    plt.ylim(np.min(w_measured[fit_t[2:-2]])*1.25, np.max(w_measured[fit_t[2:-2]])/1.25)
    plt.xlim(0, post.Lz)
    plt.ylabel('w')
    plt.legend(loc='best')
    plt.xlabel('depth')
    ax = fig.add_subplot(2,1,2)
    plt.grid(which='major')
    plt.scatter(scale_t[2:-2][this_good], np.abs(1 - (w_theory/w_measured)[this_good]), c='orange', label='theory V measured')
    plt.scatter(scale_t[2:-2][this_good], np.abs(1 - best_theory_w[2:-2][this_good]/w_measured[this_good]), c='mediumorchid', label='new theory V measured', marker='x')
    plt.ylim(1e-4, 1e0)
    plt.yscale('log')
    plt.ylabel('fractional diff')
    plt.xlabel('t')
    fig.savefig('{:s}/w_v_z.png'.format(post.full_out_dir), bbox_inches='tight', dpi=200)


    f = h5py.File('{:s}/iterative_file.h5'.format(post.full_out_dir), 'w')
    f['I0'] = best_fit_I0#i0/I0_f
    f['M0'] = best_fit_M0#p0/I0_f
    f['B']  = best_fit_B
    f['Gamma'] = best_fit_Gamma#np.mean(Gamma_fit)
    f['V0'] = best_fit_V0#np.mean(V0_fit)
    f['T0'] = best_fit_T0 
    f['beta'] = best_fit_beta#I0_f*fit_If
    f['chi'] = best_fit_chi#I0_f*fit_If
    f.close()

    for k, new, old in (('I0', best_fit_I0, fit_I0), ('M0', best_fit_M0, fit_M0), ('B', best_fit_B, fit_B),
                        ('Gamma', best_fit_Gamma, fit_Gamma), ('V0', best_fit_V0, fit_V0),
                        ('T0', best_fit_T0, fit_T0), ('beta', best_fit_beta, fit_beta), ('chi', best_fit_chi, fit_chi)):
        logger.info('key: {}, old: {:.2e}, new: {:.2e}, change: {:.2e}'.format(k, old, new, 1-new/old))

    f = h5py.File('{:s}/final_outputs.h5'.format(post.full_out_dir), 'w')
    f['good'] = good
    f['fit_t'] = fit_t
    f['times'] = scale_t
    f['B'] = int_rho_s1
    f['f'] = therm_radius/radius
    f['V0'] = V0
    f['p']  = theory_momentum
    f['I']  = theory_impulse
    f['Gamma'] = int_circ
    f['d_measured'] = d_measured
    f['d_theory'] = d_theory
    f['r_measured'] = r_measured
    f['r_theory'] = r_theory
    f['w_measured'] = w_measured
    f['w_theory'] = w_theory
    f.close()
    
    






if __name__ == '__main__':
    from docopt import docopt
    args = docopt(__doc__)
    plot = args['--plot']
    analyze = not(args['--no_analyze'])
    root_dir = args['--root_dir']
    post_process(root_dir, plot=plot, analyze=analyze, n_files=int(args['--n_files']), get_contour=args['--get_contour'], iterative=args['--iterative'])
