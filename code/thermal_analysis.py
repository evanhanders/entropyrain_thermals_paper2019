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
        post.calculate_contour()

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
    fit_Gamma = fit_file['fit_Gamma'].value     
    fit_V0    = fit_file['fit_V0'].value  
    fit_T0    = fit_file['fit_T0'].value  
    fit_If    = fit_file['fit_If'].value  

    w_cb         = fit_file['w_cb'].value
    cb_T_fit     = fit_file['cb_T_fit'].value
    vortex_T     = fit_file['vortex_T'].value
    therm_radius = fit_file['vortex_radius'].value
    height       = fit_file['vortex_height'].value
    fit_file.close()
   
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

    #Recreate thermo profiles
    height = z_measured = (vortex_T - 1)/post.grad_T_ad + post.Lz
    T0   = (1 + post.grad_T_ad*(height-post.Lz))
    rho0 = T0**post.m_ad
    rho0_z = -post.m_ad*T0**(post.m_ad-1)*post.grad_T_ad
    H_rho = T0 / post.m_ad / (-post.grad_T_ad)

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
    rho_bots = (1 + post.Lz - z_bots)**(1.5)
    rho_tops = (1 + post.Lz - z_bots - 2*Rz)**(1.5)


    #cb fit
    depth = post.Lz - height
    t_max = times.max()
    t_cond = times.max()/t_max
    times /= t_cond
    fit_t = (height < 0.75*post.Lz)*(height > 0.1*post.Lz)
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

    #Eqn1, B = \int \rho S_1 dV = const
    logger.info('plotting eqn1, integrated entropy')
    fig, axs, cb_depth_fit, (a, b) = plot_and_fit_trace(scale_t[good], int_rho_s1[good], fit_ind=fit_t[good], fit_func=linear_fit, 
                       labels=['t',r'$\int (\rho S_1) dV$'],
                       fit_str_format='{:.2g} $t$ + {:.2g}')
    axs[0].axhline(fit_B, c='green', dashes=(2,1))
    fig.savefig('{:s}/tot_entropy_v_time{:s}'.format(post.full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
    plt.close(fig)

    B_approx = scale_t*a + b

    #Eqn4, volume ~ R^3, or r^3
    logger.info('plotting eqn4, V/r^3 = V_0')
    V0 = volumes/therm_radius**3
    fig, axs, V0_fit, (dV0_dt, V0) = plot_and_fit_trace(scale_t[good*np.isfinite(V0)], V0[good*np.isfinite(V0)], fit_ind=fit_t[good*np.isfinite(V0)], fit_func=linear_fit, 
                       labels=['t',r'$V_0 = V/R^3$'],
                       fit_str_format='{:.2g} $t$ + {:.2g}')
    axs[0].axhline(fit_V0, c='green', dashes=(2,1))
    fig.savefig('{:s}/V0_v_time{:s}'.format(post.full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
    plt.close(fig)

    #Eqn 7, rho V w / B = beta t + M_0
    logger.info('plotting eqn7, rho V w / B = linear')
    p_B = rho0*volumes*w_cb/fit_B/fit_If#B_approx
    p = p_B*fit_B#B_approx
    fig, axs, cb_depth_fit, (beta, M0_div_B) = plot_and_fit_trace(scale_t[good*np.isfinite(p_B)], p_B[good*np.isfinite(p_B)], fit_ind=fit_t[good*np.isfinite(p_B)], fit_func=linear_fit, 
                       labels=['t',r'$\rho V w / B$'],
                       fit_str_format='{:.2g} $t$ + {:.2g}')
#    axs[0].axhline(fit_M0/fit_B/fit_If, c='green', dashes=(2,1))
    theory = linear_fit(post.times, fit_beta*fit_B, fit_M0)/fit_B
    axs[0].plot(post.times, theory, c='indigo')
    fig.savefig('{:s}/momentum_div_B_v_time{:s}'.format(post.full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
    plt.close(fig)

    #Eqn 8, 0.5*rho0*r^2*Gamma / B = t + Const
    logger.info('plotting eqn8, 0.5* rho r^2 Gamma / B = t + const')
    I_B = np.pi*rho0*therm_radius**2*int_circ/fit_B/fit_If#B_approxa
    I_B_true = int_impulse/fit_B
    I = I_B*fit_B#B_approx
    theory = linear_fit(post.times, fit_B, fit_I0)/fit_B
    fig, axs, I_B_fit, (I0_f, I0_div_B) = plot_and_fit_trace(scale_t[good*np.isfinite(I_B)], I_B[good*np.isfinite(I_B)], fit_ind=fit_t[good*np.isfinite(I_B)], fit_func=linear_fit, 
                       labels=['t',r'$\pi\rho r^2 \Gamma / B$'],
                       fit_str_format='{:.2g} $t$ + {:.2g}')
    axs[0].plot(post.times, theory, c='indigo')
#    axs[0].axhline(fit_I0/fit_B/fit_If, c='green', dashes=(2,1))
    fig.savefig('{:s}/impulse_div_B_v_time{:s}'.format(post.full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
    plt.close(fig)

    #Eqn9, B = \int omega dA = const
    logger.info('plotting eqn9, circulation')
    fig, axs, Gamma_fit, (a, b) = plot_and_fit_trace(scale_t[good],int_circ[good], fit_ind=fit_t[good], fit_func=linear_fit, 
                       labels=['t',r'$\Gamma_{thermal}$'],
                       fit_str_format='{:.2g} $t$ + {:.2g}')
    print(fit_Gamma)
    axs[0].axhline(fit_Gamma, c='green', dashes=(2,1))
    fig.savefig('{:s}/circulation_v_time{:s}'.format(post.full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
    plt.close(fig)


    #Eqn 10, r = sqrt ( 2 * [Bt + I_0] / rho / Gamma )
    logger.info('plotting eqn10, radius fit')
    r_fit = np.sqrt((fit_If)*(fit_B*scale_t + fit_I0) / rho0 / fit_Gamma / np.pi)
#    r_fit = np.sqrt((fit_B)*(I0_f*scale_t + I0_div_B) / rho0 / int_circ / np.pi)

    fig = plt.figure()
    ax = fig.add_subplot(2,1,1) 
    plt.plot(scale_t[good], therm_radius[good], label='r')
    plt.plot(scale_t[good], r_fit[good], label=r'$\sqrt{{\frac{{(Bt + I_0)}}{{\pi \rho \Gamma}}}}$')
#    plt.plot(scale_t[good], r_fit[good], label=r'$\sqrt{{\frac{{({:.2f}Bt + I_0)}}{{\pi \rho \Gamma}}}}$'.format(I0_f))
    plt.ylim(0, 1.25*np.max(therm_radius))
    plt.ylabel('r')
    plt.legend(loc='best')
    ax = fig.add_subplot(2,1,2)
    plt.plot(scale_t[good], np.abs(1 - therm_radius[good]/r_fit[good]))
    plt.yscale('log')
    plt.ylabel('fractional diff')
    plt.xlabel('t')
    fig.savefig('{:s}/r_Fit_v_time.png'.format(post.full_out_dir), bbox_inches='tight', dpi=200)


    #Eqn 11, 
    logger.info('plotting eqn 4 & 11, momentum fit')
    V_est = fit_V0 * r_fit**3
    momentum_est = rho0 * V_est * w_cb
    momentum_linear = fit_B*fit_If*(beta*scale_t+M0_div_B)

    fig = plt.figure()
    ax = fig.add_subplot(2,1,1) 
    plt.plot(scale_t[good], int_mom[good], c='k', label=r'$\int \rho w dV$')
    plt.plot(scale_t[good], momentum_est[good], c='indigo', label=r'$\rho V_0 r_{fit}^3 w_{cb}$')
    plt.plot(scale_t[good], momentum_linear[good], c='orange', ls='--', label=r'$B\left(\beta t + \frac{M_0}{B}\right)$')
    if np.mean(B_approx) < 0:
        plt.ylim(int_mom[good].min()*1.25, 0)
    else:
        plt.ylim(0, int_mom[good].max()*1.25)

    plt.ylabel('momentum')
    plt.legend(loc='best')
    ax = fig.add_subplot(2,1,2)
    plt.grid(which='major')
    plt.plot(scale_t[good], np.abs(1 - momentum_est[good]/int_mom[good]), c='indigo', label='theory V measured')
    plt.plot(scale_t[good], np.abs(1 - momentum_linear[good]/int_mom[good]), c='orange', label='linear V measured')
    plt.ylim(1e-4, 1e0)
    plt.yscale('log')
    plt.ylabel('fractional diff')
    plt.xlabel('t')
    fig.savefig('{:s}/p_Fit_v_time.png'.format(post.full_out_dir), bbox_inches='tight', dpi=200)

    # Entrainment rate, dlnV_dz
    logger.info('plotting dlnV_dz = dlnV_dt / w_cb')
    lnV = np.log(volumes)
    dlnV_dt = np.diff(lnV)/np.diff(scale_t)
    eps = - dlnV_dt / w_cb[1:]
    use = good[1:]*np.isfinite(eps)*(scale_t[1:]>2)*fit_t[1:]
    fig, axs, cb_depth_fit, (a, b) = plot_and_fit_trace(scale_t[1:][use], eps[use], fit_ind=fit_t[1:][use], fit_func=linear_fit, 
                       labels=['t',r'$\frac{d\ln V}{dz}$'],
                       fit_str_format='{:.2g} $t$ + {:.2g}')
    fig.savefig('{:s}/entrainment_rate_v_time{:s}'.format(post.full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
    plt.close(fig)

    #Outputs: z(t), r(z), w(z)
    logger.info('plotting z(t)')
    z_theory   = ((cb_T_fit - 1)/post.grad_T_ad + post.Lz)
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
    plt.plot(scale_t[good], np.abs(1 - d_theory[good]/d_measured[good]), c='orange', label='theory V measured')
    plt.ylim(1e-4, 1e0)
    plt.yscale('log')
    plt.ylabel('fractional diff')
    plt.xlabel('t')

    fig.savefig('{:s}/z_v_t.png'.format(post.full_out_dir), bbox_inches='tight', dpi=200)



    logger.info('plotting r(z)')
    r_measured = therm_radius
    r_theory   = r_fit
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1) 
    plt.scatter(d_measured[good], r_measured[good], c='k', label=r'$r$ (measured)', marker='+')
    plt.plot(d_theory[good],   r_theory[good], c='orange', label=r'$r$ (theory)')
    plt.ylim(0, np.max(r_measured)*1.25)
    plt.xlim(0, post.Lz)
    plt.ylabel('r')
    plt.legend(loc='best')
    plt.xlabel('depth')
    fig.savefig('{:s}/r_v_z.png'.format(post.full_out_dir), bbox_inches='tight', dpi=200)

    logger.info('plotting w(z)')
    w_measured = differentiate(scale_t, z_measured)
    w_theory   = w_cb[2:-2]
    this_good  = good[2:-2]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1) 
    plt.scatter(d_measured[2:-2][this_good], w_measured[this_good], c='k', marker='+', label=r'$w$ (measured)')
    plt.plot(d_theory[2:-2][this_good],  w_theory[this_good],   c='orange', label=r'$w$ (theory)')
    plt.ylim(np.min(w_measured[fit_t[2:-2]])*1.25, np.max(w_measured[fit_t[2:-2]])/1.25)
    plt.xlim(0, post.Lz)
    plt.ylabel('w')
    plt.legend(loc='best')
    plt.xlabel('depth')
    fig.savefig('{:s}/w_v_z.png'.format(post.full_out_dir), bbox_inches='tight', dpi=200)


    logger.info('plotting integrated / approx quantities')
    impulse = int_impulse
    momentum = int_mom
    momentum_approx = p_B * fit_B * fit_If#B_approx
    impulse_approx  = I_B * fit_B * fit_If#B_approx
    measured_If     = impulse_approx/impulse
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1) 
    plt.grid()
    plt.plot(scale_t[good], momentum_approx[good]/momentum[good], c='k', label=r'$(\rho V w) / P_z$')
    plt.plot(scale_t[good], impulse_approx[good]/impulse[good],   c='orange', label=r'$(\pi\rho r^2\Gamma)/I_z$')
    plt.ylim(0.5, 1.5)
    plt.ylabel('fractions')
    plt.legend(loc='best')
    plt.xlabel('t')
    fig.savefig('{:s}/fractions_v_t.png'.format(post.full_out_dir), bbox_inches='tight', dpi=200)


    f = h5py.File('{:s}/iterative_file.h5'.format(post.full_out_dir), 'w')
    (a_i, i0), pcov = scop.curve_fit(linear_fit, scale_t[fit_t], I[fit_t])
    (a_p, p0), pcov = scop.curve_fit(linear_fit, scale_t[fit_t], p[fit_t])
    print(a_i, a_p, I0_f*fit_If)
    f['I0'] = i0/I0_f
    f['M0'] = p0/I0_f
    f['B']  = np.mean(B_approx)
    f['Gamma'] = np.mean(Gamma_fit)
    f['V0'] = np.mean(V0_fit)
    f['T0'] = fit_T0 
    f['If'] = I0_f*fit_If
    f.close()

    f = h5py.File('{:s}/final_outputs.h5'.format(post.full_out_dir), 'w')
    f['good'] = good
    f['fit_t'] = fit_t
    f['times'] = scale_t
    f['B'] = int_rho_s1
    f['B_approx'] = B_approx 
    f['f'] = therm_radius/radius
    f['V0'] = V0
    f['If'] = measured_If
    f['p']  = p
    f['I']  = I
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
