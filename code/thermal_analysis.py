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
    avg_ke_flux  = output_file['int_ke_flux'].value/volumes
    avg_enth_flux  = output_file['int_enth_flux'].value/volumes
    output_file.close()

    cb_file           = h5py.File('{:s}/z_cb_file.h5'.format(post.full_out_dir), 'r')
    T0 = vortex_T     = cb_file['vortex_T'].value
    vortex_w          = cb_file['vortex_w'].value
    rho0 = vortex_rho = cb_file['vortex_rho'].value
    therm_radius      = cb_file['vortex_radius'].value
    height            = cb_file['vortex_height'].value
    cb_file.close()
    height = z_measured = (T0 - 1)/post.grad_T_ad + post.Lz
  
    #Read in contours
    contour_file = h5py.File('{:s}/contour_file.h5'.format(post.full_out_dir), 'r')
    z        = contour_file['z'].value
    contours = contour_file['contours'].value
    contour_file.close()

    thermal_found = np.zeros(contours.shape[0], dtype =bool) 
    for i in range(contours.shape[0]):
        if not (contours[i,-1] != 0)*(contours[i,-2]==0):
            thermal_found[i] = True
    good = thermal_found*(height > 0.25*post.Lz)

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
    fit_t = (height < 0.65*post.Lz)*(height > 0.25*post.Lz)
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
    (dGamma, Gamma0), pcov = scop.curve_fit(linear_fit, scale_t[fit_t], int_circ[fit_t])
    fit_B0 = B0
    fit_Gamma0 = Gamma0 + dGamma*scale_t[fit_t][-1]/2

#    I_B = int_impulse/fit_B0
    I_B = rho0*np.pi*therm_radius**2*fit_Gamma0/fit_B0
    (chi, I0_div_B), pcov = scop.curve_fit(linear_fit, scale_t[fit_t], I_B[fit_t])
    fit_chi = chi

#    p_B = int_mom/fit_B0/fit_chi
#    (beta, M0_div_B), pcov = scop.curve_fit(linear_fit, scale_t[fit_t], p_B[fit_t])
#    fit_beta = beta 

    I_Iz = (int_impulse / (rho0 * volumes))[2:-2] / vortex_w
    (slope, beta_plus_one), pcov = scop.curve_fit(linear_fit, scale_t[2:-2][fit_t[2:-2]], I_Iz[fit_t[2:-2]])
    fit_beta = beta_plus_one - 1
    print(fit_beta, 'fit beta')



    f = volumes/therm_radius**3
    (dfdt, f0), pcov = scop.curve_fit(linear_fit, scale_t[fit_t], f[fit_t])

    this_fit = good*post.get_good_times(height, L_max=0.5, L_min=0.1)

    print(scale_t[this_fit][0])
    f_factor = np.mean((I_B / (int_impulse/fit_B0))[this_fit])
    bounds = (-scale_t[this_fit][0]/3, scale_t[this_fit][0]/3)
    p0     = (0)
    this_r_theory = lambda t, t_off: theory_r(t, fit_B0, fit_Gamma0, fit_chi, t_off, rho_f=interp1d(scale_t, rho0))
    fit, pcov = scop.curve_fit(this_r_theory, scale_t[this_fit], therm_radius[this_fit], bounds=bounds, p0=p0, maxfev=1e4)
    fit_t_off = fit[0]
    scale_t += fit_t_off    

    bounds = ((0,                        0.5*f0), 
              (1 + (vortex_T.max()-1)/2, 1.5*f0))
    p0 = (1, f0)
    this_T_theory      = lambda times, T, f:  theory_T(times, fit_B0, fit_Gamma0, f, fit_chi, fit_beta, 0, T, grad_T_ad=post.grad_T_ad, m_ad=post.m_ad)
    (fit_T0, fit_f), pcov = scop.curve_fit(this_T_theory, scale_t[this_fit], T0[this_fit], bounds=bounds, p0=p0, maxfev=1e4)

    theory_impulse  = linear_fit(scale_t, fit_B0*fit_chi,   0)
    theory_momentum = linear_fit(scale_t, fit_B0*fit_chi*fit_beta, 0)
    fit_T           = this_T_theory(scale_t, fit_T0, fit_f)
    theory_w        = theory_dT_dt(scale_t, fit_B0, fit_Gamma0, fit_f, fit_chi, fit_beta, 0, fit_T0, grad_T_ad=post.grad_T_ad, m_ad=post.m_ad)/post.grad_T_ad
    theory_radius   = this_r_theory(scale_t, 0)#np.sqrt(theory_impulse/(np.pi*Gamma0*fit_T**(post.m_ad)))

    #Eqn1, B = \int \rho S_1 dV = const
    logger.info('plotting eqn1, integrated entropy')
    fig, axs, B_fit, (a, B0) = plot_and_fit_trace(scale_t[good], int_rho_s1[good], fit_ind=fit_t[good], fit_func=linear_fit, 
                       labels=['t',r'$\int (\rho S_1) dV$'],
                       fit_str_format='{:.2g} $t$ + {:.2g}')
    axs[0].axhline(fit_B0, c='green', dashes=(2,1))
    logger.info('entropy frac change: {}'.format(a*scale_t[fit_t][-1]/B0))
    fig.savefig('{:s}/tot_entropy_v_time{:s}'.format(post.full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
    plt.close(fig)


    #Eqn4, volume ~ R^3, or r^3
    logger.info('plotting eqn4, V/r^3 = V_0')
    f = volumes/therm_radius**3
    fig, axs, f_fit, (df_dt, f) = plot_and_fit_trace(scale_t[good*np.isfinite(f)], f[good*np.isfinite(f)], fit_ind=fit_t[good*np.isfinite(f)], fit_func=linear_fit, 
                       labels=['t',r'$V_0 = V/r^3$'],
                       fit_str_format='{:.2g} $t$ + {:.2g}')
    axs[0].set_ylim(5, 15)
    axs[0].axhline(fit_f, c='green', dashes=(2,1))
    axs[0].axhline(f_factor**(3./2)*fit_f, c='red', dashes=(2,1))
    fig.savefig('{:s}/f_v_time{:s}'.format(post.full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
    plt.close(fig)

    #Eqn 7, rho V w / B = beta t + M_0
    logger.info('plotting eqn7, rho V w / B = linear')
    p_B = int_mom/fit_B0/fit_beta/fit_chi
    fig, axs, momentum_div_B, (beta, M0_div_B) = plot_and_fit_trace(scale_t[good*np.isfinite(p_B)], p_B[good*np.isfinite(p_B)], fit_ind=fit_t[good*np.isfinite(p_B)], fit_func=linear_fit, 
                       labels=['t',r'$\int \rho w dV / B / (\beta\chi)$'],
                       fit_str_format='{:.2g} $t$ + {:.2g}')
    axs[0].plot(post.times, theory_momentum/fit_B0/(fit_beta)/fit_chi, c='green', dashes=(2,1))
    fig.savefig('{:s}/momentum_div_B_v_time{:s}'.format(post.full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
    plt.close(fig)

    #Eqn9, B = \int omega dA = const
    logger.info('plotting eqn9, circulation')
    fig, axs, Gamma_fit, (a, Gamma0) = plot_and_fit_trace(scale_t[good],int_circ[good], fit_ind=fit_t[good], fit_func=linear_fit, 
                       labels=['t',r'$\Gamma_{thermal}$'],
                       fit_str_format='{:.2g} $t$ + {:.2g}')
    axs[0].axhline(fit_Gamma0, c='green', dashes=(2,1))
    fig.savefig('{:s}/circulation_v_time{:s}'.format(post.full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
    plt.close(fig)

    #Eqn 8, 0.5*rho0*r^2*Gamma / B = t + Const
    logger.info('plotting eqn8, 0.5* rho r^2 Gamma / B = t + const')
    I_B = np.pi*therm_radius**2*fit_Gamma0*rho0/fit_B0/fit_chi
#    I_B = int_impulse/fit_B/fit_beta
    fig, axs, I_B_fit, (I0_f, I0_div_B) = plot_and_fit_trace(scale_t[good*np.isfinite(I_B)], I_B[good*np.isfinite(I_B)], fit_ind=fit_t[good*np.isfinite(I_B)], fit_func=linear_fit, 
                       labels=['t',r'$\pi\rho r^2 \Gamma / B / \beta$'],
                       fit_str_format='{:.2g} $t$ + {:.2g}')
    axs[0].plot(post.times, theory_impulse/fit_B0/fit_chi, c='green', dashes=(2,1))
    fig.savefig('{:s}/impulse_div_B_v_time{:s}'.format(post.full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
    plt.close(fig)

    #Outputs: z(t), r(z), w(z)
    logger.info('plotting z(t)')
    z_theory   = ((fit_T - 1)/post.grad_T_ad + post.Lz)
    z_measured = ((vortex_T - 1)/post.grad_T_ad + post.Lz)

    d_theory   = post.Lz - z_theory
    d_measured = post.Lz - z_measured
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1) 
    plt.scatter(scale_t[good], d_measured[good], c='k', label=r'$L_z - z$ (measured)', marker='+')
    plt.plot(scale_t[good], d_theory[good], c='orange', label=r'$L_z - z$ (theory)')
    plt.ylabel('depth')
    plt.legend(loc='best')
    plt.ylim(0, post.Lz)
    ax = fig.add_subplot(2,1,2)
    plt.grid(which='major')
    plt.scatter(scale_t[good], np.abs(1 - d_theory[good]/d_measured[good]), c='orange', label='theory V measured')
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
    plt.ylim(np.min(r_measured[good])*0.75, np.max(r_measured[good])*1.25)
    plt.xlim(0, post.Lz)
    plt.ylabel('r')
    plt.legend(loc='best')
    plt.xlabel('depth')
    ax = fig.add_subplot(2,1,2)
    plt.grid(which='major')
    plt.scatter(scale_t[good], np.abs(1 - r_theory[good]/r_measured[good]), c='orange', label='theory V measured')
    plt.ylim(1e-4, 1e0)
    plt.yscale('log')
    plt.ylabel('fractional diff')
    plt.xlabel('t')
    fig.savefig('{:s}/r_v_z.png'.format(post.full_out_dir), bbox_inches='tight', dpi=200)

    logger.info('plotting w(z)')
    w_measured = vortex_w#differentiate(scale_t, z_measured)
    w_m_2      = np.diff(z_measured)/np.diff(scale_t)
    w_theory   = theory_w[2:-2]
    this_good  = good[2:-2]
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1) 
    plt.scatter(d_measured[2:-2][this_good], w_measured[this_good], c='k', marker='+', label=r'$w$ (measured)')
    plt.plot(d_theory[2:-2][this_good],  w_theory[this_good],   c='orange', label=r'$w$ (theory)')
    plt.ylim(np.min(w_measured[fit_t[2:-2]])*1.25, np.max(w_measured[fit_t[2:-2]])/1.25)
    plt.xlim(0, post.Lz)
    plt.ylabel('w')
    plt.legend(loc='best')
    plt.xlabel('depth')
    ax = fig.add_subplot(2,1,2)
    plt.grid(which='major')
    plt.scatter(scale_t[2:-2][this_good], np.abs(1 - (w_theory/w_measured)[this_good]), c='orange', label='theory V measured')
    plt.ylim(1e-4, 1e0)
    plt.yscale('log')
    plt.ylabel('fractional diff')
    plt.xlabel('t')
    fig.savefig('{:s}/w_v_z.png'.format(post.full_out_dir), bbox_inches='tight', dpi=200)


    logger.info('plotting fluxes')
    fig = plt.figure()

    plt.plot(d_measured, avg_ke_flux, c='b')
    plt.plot(d_measured, avg_enth_flux, c='r')
    plt.plot(d_measured, avg_enth_flux + avg_ke_flux, c='k')
    plt.xlabel('depth')
    plt.ylabel('avg flux')
    plt.savefig('{:s}/fluxes_v_depth.png'.format(post.full_out_dir), bbox_inches='tight', dpi=200)


    f = h5py.File('{:s}/fit_file.h5'.format(post.full_out_dir), 'w')
    f['B0']       = fit_B0
    f['Gamma0']   = fit_Gamma0
    f['f']        = fit_f
    f['T0']       = fit_T0 
    f['beta']     = fit_beta
    f['chi']      = fit_chi
    f['fit_w']    = theory_w
    f['fit_T']    = fit_T
    f['fit_r']    = theory_radius
    f['fit_z']    = ((fit_T) - 1)/post.grad_T_ad + post.Lz
    f['fit_d']    = -((fit_T) - 1)/post.grad_T_ad
    f.close()

    for k, fit in (('B0', fit_B0), ('Gamma0', fit_Gamma0), ('f', fit_f),
                        ('T0', fit_T0), ('beta', fit_beta), ('chi', fit_chi),
                        ('t_off', fit_t_off)):
        logger.info('key: {}, fit: {:.2e}'.format(k, fit))

    logger.info('adjusted f: {}'.format(fit_f*f_factor**(3./2)))

    f = h5py.File('{:s}/final_outputs.h5'.format(post.full_out_dir), 'w')
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
