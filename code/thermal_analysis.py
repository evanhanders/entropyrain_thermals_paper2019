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
    full_out_dir, n_rho, aspect, gamma, m_ad, Cp, Lz, Lr, grad_T_ad, g, files, prof_files, L_factor, t_factor, s_factor, twoD = get_basic_run_info(root_dir, out_dir)

    count           = 0
    times           = []
    for fn, fp in zip(files, prof_files):
        logger.info('reading time on file {:s}'.format(fn))
        f = h5py.File(fp, 'r')
        these_times = f['scales']['sim_time'].value
        [times.append(t) for t in these_times]
    times = np.array(times)*t_factor

    t0 = times[0]
    times -= t0

    #### Step 2.
    #If flagged, solve thermal tracking algorithm to get contour
    if get_contour:
        measure_cb(root_dir, out_dir, times)
        fit_w_cb(root_dir, out_dir, times, iterative=iterative)
        calculate_contour(root_dir, out_dir, times)

    #### Step 3.
    #If flagged, plot colormeshes
    if plot:
        plot_colormeshes(root_dir, out_dir, times)
    
    #### Step 4.
    #If the thermal contour is found, this section calculates quantities inside (or outside) of the thermal from sim outputs
    if analyze:
        measure_values_in_thermal(root_dir, out_dir, times)

    #### Step 5.
    #Read in post-processed data and make plots

    output_file = h5py.File('{:s}/post_analysis.h5'.format(full_out_dir), 'r')
    times       = output_file['times'].value
    int_circ    = output_file['int_circ'].value
    int_circ_a  = output_file['int_circ_above'].value
    int_s1      = output_file['int_s1'].value
    int_mass    = output_file['int_mass'].value
    int_impulse = output_file['int_impulse'].value
    int_impulse_p1 = output_file['int_impulse_p1'].value
    impulse_piece1 = output_file['impulse_piece1'].value
    int_impulse_p2 = output_file['int_impulse_p2'].value
    int_ke      = output_file['int_ke'].value
    int_rho_s1_a= output_file['int_rho_s1_above'].value
    int_rho_s1  = output_file['int_rho_s1'].value
    int_mom     = output_file['int_mom'].value
    int_pe      = output_file['int_pe'].value
    int_w       = output_file['int_w'].value
    int_u       = output_file['int_u'].value
    volumes     = output_file['int_vol'].value
    area        = output_file['int_area'].value
    radius      = output_file['radius'].value
    w_cb        = output_file['w_cb'].value
    z_cb        = output_file['z_cb'].value
    max_s1_therm = output_file['max_s1_therm'].value
    torus_vol   = output_file['torus_vol'].value
    full_impulse   = output_file['full_impulse'].value
    full_momentum   = output_file['full_momentum'].value
    full_ke    = output_file['full_ke'].value

    output_file.close()




    output_file = h5py.File('{:s}/fit_file.h5'.format(full_out_dir), 'r')
    fit_B   = output_file['fit_B']           .value 
    fit_M0   = output_file['fit_M0']         .value  
    fit_I0   = output_file['fit_I0']         .value  
    fit_beta   = output_file['fit_beta']     .value    
    fit_Gamma   = output_file['fit_Gamma']   .value     
    fit_V0   = output_file['fit_V0']         .value  
    fit_T0   = output_file['fit_T0']         .value  

    w_cb = output_file['w_cb'].value
    cb_T_fit = output_file['cb_T_fit'].value
    vortex_T = output_file['vortex_T'].value
    therm_radius = output_file['vortex_radius'].value
    height      = output_file['vortex_height'].value
    output_file.close()
    output_file = h5py.File('{:s}/contour_file.h5'.format(full_out_dir), 'r')
    z = output_file['z'].value
    contours = output_file['contours'].value
    output_file.close()

    thermal_found = np.zeros(contours.shape[0], dtype =bool) 
    for i in range(contours.shape[0]):
        if not (contours[i,-1] != 0)*(contours[i,-2]==0):
            thermal_found[i] = True
    good = thermal_found

    #Recreate thermo profiles
    height = z_measured = (vortex_T - 1)/grad_T_ad + Lz
    T0   = (1 + grad_T_ad*(height-Lz))
    rho0 = T0**m_ad
    rho0_z = -m_ad*T0**(m_ad-1)*grad_T_ad
    H_rho = T0 / m_ad / (-grad_T_ad)

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
    rho_bots = (1 + Lz - z_bots)**(1.5)
    rho_tops = (1 + Lz - z_bots - 2*Rz)**(1.5)


    #cb fit
    depth = Lz - z_cb
    t_max = times.max()
    t_cond = times.max()/t_max
    times /= t_cond
    fit_t = (height < 0.7*Lz)*(height > 0.1*Lz)
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
    fig.savefig('{:s}/tot_entropy_v_time{:s}'.format(full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
    plt.close(fig)

    B_approx = scale_t*a + b

    #Eqn2, Aspect ratio = A = R/R_z = const
    logger.info('plotting eqn2, aspect ratio')
    aspect_ratio = therm_radius/(Rz)
    aspect_ratio[np.isinf(aspect_ratio)] = 0
    aspect_ratio[np.isnan(aspect_ratio)] = 0
    fig, axs, cb_depth_fit, (a, b) = plot_and_fit_trace(scale_t[good], aspect_ratio[good], fit_ind=fit_t[good], fit_func=linear_fit, 
                       labels=['t',r'$A = R / R_z$'],
                       fit_str_format='{:.2g} $t$ + {:.2g}')
    axs[0].axhline(4*np.pi/3/fit_V0, c='green', dashes=(2,1))
    fig.savefig('{:s}/aspect_ratio_v_time{:s}'.format(full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
    plt.close(fig)

#    #Eqn3, r = f R, vortex core radius / total radius is a fraction and constant
#    logger.info('plotting eqn3, r = f R')
#    f = therm_radius/radius
#    fig, axs, f_fit, (df_dt, f_Fit) = plot_and_fit_trace(scale_t[good*np.isfinite(f)], f[good*np.isfinite(f)], fit_ind=fit_t[good*np.isfinite(f)], fit_func=linear_fit, 
#                       labels=['t',r'$f = r/R$'],
#                       fit_str_format='{:.2g} $t$ + {:.2g}')
#    axs[0].axhline(fit_f, c='green', dashes=(2,1))
#    fig.savefig('{:s}/rR_fraction_v_time{:s}'.format(full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
#    plt.close(fig)

    #Eqn4, volume ~ R^3, or r^3
    logger.info('plotting eqn4, V/r^3 = V_0')
    V0 = volumes/therm_radius**3
    fig, axs, V0_fit, (dV0_dt, V0) = plot_and_fit_trace(scale_t[good*np.isfinite(V0)], V0[good*np.isfinite(V0)], fit_ind=fit_t[good*np.isfinite(V0)], fit_func=linear_fit, 
                       labels=['t',r'$V_0 = V/R^3$'],
                       fit_str_format='{:.2g} $t$ + {:.2g}')
    axs[0].axhline(fit_V0, c='green', dashes=(2,1))
    fig.savefig('{:s}/V0_v_time{:s}'.format(full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
    plt.close(fig)

    #Eqn 7, rho V w / B = beta t + M_0
    logger.info('plotting eqn7, rho V w / B = linear')
    p_B = rho0*volumes*w_cb/B_approx
    p = p_B*B_approx
    fig, axs, cb_depth_fit, (beta, M0_div_B) = plot_and_fit_trace(scale_t[good*np.isfinite(p_B)], p_B[good*np.isfinite(p_B)], fit_ind=fit_t[good*np.isfinite(p_B)], fit_func=linear_fit, 
                       labels=['t',r'$\rho V w / B$'],
                       fit_str_format='{:.2g} $t$ + {:.2g}')
    axs[0].axhline(fit_M0/B_approx[0], c='green', dashes=(2,1))
    fig.savefig('{:s}/momentum_div_B_v_time{:s}'.format(full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
    plt.close(fig)

    #Eqn 8, 0.5*rho0*r^2*Gamma / B = t + Const
    logger.info('plotting eqn8, 0.5* rho r^2 Gamma / B = t + const')
    I_B = np.pi*rho0*therm_radius**2*int_circ/B_approx
    I = I_B*B_approx
    fig, axs, I_B_fit, (I0_f, I0_div_B) = plot_and_fit_trace(scale_t[good*np.isfinite(I_B)], I_B[good*np.isfinite(I_B)], fit_ind=fit_t[good*np.isfinite(I_B)], fit_func=linear_fit, 
                       labels=['t',r'$\pi\rho r^2 \Gamma / B$'],
                       fit_str_format='{:.2g} $t$ + {:.2g}')
    axs[0].axhline(fit_I0/B_approx[0], c='green', dashes=(2,1))
    fig.savefig('{:s}/impulse_div_B_v_time{:s}'.format(full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
    plt.close(fig)

    #Eqn9, B = \int omega dA = const
    logger.info('plotting eqn9, circulation')
    fig, axs, Gamma_fit, (a, b) = plot_and_fit_trace(scale_t[good],int_circ[good], fit_ind=fit_t[good], fit_func=linear_fit, 
                       labels=['t',r'$\Gamma_{thermal}$'],
                       fit_str_format='{:.2g} $t$ + {:.2g}')
    axs[0].axhline(fit_I0/fit_Gamma, c='green', dashes=(2,1))
    print(fit_Gamma)
    fig.savefig('{:s}/circulation_v_time{:s}'.format(full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
    plt.close(fig)


    #Eqn 10, r = sqrt ( 2 * [Bt + I_0] / rho / Gamma )
    logger.info('plotting eqn10, radius fit')
    r_fit = np.sqrt((B_approx)*(I0_f*scale_t + I0_div_B) / rho0 / int_circ / np.pi)
    print(r_fit)

    fig = plt.figure()
    ax = fig.add_subplot(2,1,1) 
    plt.plot(scale_t[good], therm_radius[good], label='r')
    plt.plot(scale_t[good], r_fit[good], label=r'$\sqrt{{\frac{{({:.2f}Bt + I_0)}}{{\pi \rho \Gamma}}}}$'.format(I0_f))
    plt.ylim(0, 1.25*np.max(therm_radius))
    plt.ylabel('r')
    plt.legend(loc='best')
    ax = fig.add_subplot(2,1,2)
    plt.plot(scale_t[good], np.abs(1 - therm_radius[good]/r_fit[good]))
    plt.yscale('log')
    plt.ylabel('fractional diff')
    plt.xlabel('t')
    fig.savefig('{:s}/r_Fit_v_time.png'.format(full_out_dir), bbox_inches='tight', dpi=200)


    #Eqn 11, 
    logger.info('plotting eqn 4 & 11, momentum fit')
    V_est = V0 * r_fit**3
#    w_cb  = (z_cb[2:] - z_cb[:-2])/(scale_t[2:] - scale_t[:-2])
    momentum_est = rho0 * V_est * w_cb
    momentum_linear = B_approx*(beta*scale_t+M0_div_B)

    fig = plt.figure()
    ax = fig.add_subplot(2,1,1) 
    plt.plot(scale_t[good], int_mom[good], c='k', label=r'$\int \rho w dV$')
    plt.plot(scale_t[good], momentum_est[good], c='indigo', label=r'$\rho\frac{V_0}{f^3}r_{fit}^3 \frac{d z_{cb}}{dt}$')
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
    fig.savefig('{:s}/p_Fit_v_time.png'.format(full_out_dir), bbox_inches='tight', dpi=200)

    # Entrainment rate, dlnV_dz
    logger.info('plotting dlnV_dz = dlnV_dt / w_cb')
    lnV = np.log(volumes)
    dlnV_dt = np.diff(lnV)/np.diff(scale_t)
    eps = - dlnV_dt / w_cb[1:]
    use = good[1:]*np.isfinite(eps)*(scale_t[1:]>2)*fit_t[1:]
    fig, axs, cb_depth_fit, (a, b) = plot_and_fit_trace(scale_t[1:][use], eps[use], fit_ind=fit_t[1:][use], fit_func=linear_fit, 
                       labels=['t',r'$\frac{d\ln V}{dz}$'],
                       fit_str_format='{:.2g} $t$ + {:.2g}')
    fig.savefig('{:s}/entrainment_rate_v_time{:s}'.format(full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
    plt.close(fig)

    #Outputs: z(t), r(z), w(z)
    logger.info('plotting z(t)')
    z_theory   = ((cb_T_fit - 1)/grad_T_ad + Lz)
    z_measured = ((vortex_T - 1)/grad_T_ad + Lz)

    d_theory   = Lz - z_theory
    d_measured = Lz - z_measured
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1) 
    plt.scatter(scale_t[good], d_measured[good], c='k', label=r'$L_z - z$ (measured)', marker='+')
    plt.plot(scale_t[good], d_theory[good], c='orange', label=r'$L_z - z$ (theory)')
    plt.ylabel('depth')
    plt.legend(loc='best')
    plt.ylim(0, Lz)
    ax = fig.add_subplot(2,1,2)
    plt.grid(which='major')
    plt.plot(scale_t[good], np.abs(1 - d_theory[good]/d_measured[good]), c='orange', label='theory V measured')
    plt.ylim(1e-4, 1e0)
    plt.yscale('log')
    plt.ylabel('fractional diff')
    plt.xlabel('t')

    fig.savefig('{:s}/z_v_t.png'.format(full_out_dir), bbox_inches='tight', dpi=200)



    logger.info('plotting r(z)')
    r_measured = therm_radius
    r_theory   = r_fit
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1) 
    plt.scatter(d_measured[good], r_measured[good], c='k', label=r'$r$ (measured)', marker='+')
    plt.plot(d_theory[good],   r_theory[good], c='orange', label=r'$r$ (theory)')
    plt.ylim(0, np.max(r_measured)*1.25)
    plt.xlim(0, Lz)
    plt.ylabel('r')
    plt.legend(loc='best')
    plt.xlabel('depth')
    fig.savefig('{:s}/r_v_z.png'.format(full_out_dir), bbox_inches='tight', dpi=200)

    logger.info('plotting w(z)')
    w_measured = differentiate(scale_t, z_measured)
    w_theory   = w_cb[2:-2]
    this_good  = good[2:-2]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1) 
    plt.scatter(d_measured[2:-2][this_good], w_measured[this_good], c='k', marker='+', label=r'$w$ (measured)')
    plt.plot(d_theory[2:-2][this_good],  w_theory[this_good],   c='orange', label=r'$w$ (theory)')
    plt.ylim(np.min(w_measured[fit_t[2:-2]])*1.25, np.max(w_measured[fit_t[2:-2]])/1.25)
    plt.xlim(0, Lz)
    plt.ylabel('w')
    plt.legend(loc='best')
    plt.xlabel('depth')
    fig.savefig('{:s}/w_v_z.png'.format(full_out_dir), bbox_inches='tight', dpi=200)


    logger.info('plotting integrated / approx quantities')
    impulse = int_impulse
    momentum = int_mom
    momentum_approx = p_B * B_approx
    impulse_approx  = I_B * B_approx
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1) 
    plt.grid()
    plt.plot(scale_t[good], momentum_approx[good]/momentum[good], c='k', label=r'$(\rho V w) / P_z$')
    plt.plot(scale_t[good], impulse_approx[good]/impulse[good],   c='orange', label=r'$(\pi\rho r^2\Gamma)/I_z$')
    plt.ylim(0.5, 1.5)
    plt.ylabel('fractions')
    plt.legend(loc='best')
    plt.xlabel('t')
    fig.savefig('{:s}/fractions_v_t.png'.format(full_out_dir), bbox_inches='tight', dpi=200)



    f = h5py.File('{:s}/iterative_file.h5'.format(full_out_dir), 'w')
    (a_i, i0), pcov = scop.curve_fit(linear_fit, scale_t[fit_t], I[fit_t])
    (a_p, p0), pcov = scop.curve_fit(linear_fit, scale_t[fit_t], p[fit_t])
    print(a_i, a_p)
    f['I0'] = i0
    f['M0'] = p0
    f['B']  = np.mean(B_approx)
    f['Gamma'] = np.mean(Gamma_fit)
    f['V0'] = np.mean(V0_fit)
    f.close()

    f = h5py.File('{:s}/final_outputs.h5'.format(full_out_dir), 'w')
    f['good'] = good
    f['fit_t'] = fit_t
    f['times'] = scale_t
    f['B'] = int_rho_s1
    f['B_approx'] = B_approx 
    f['A'] = aspect_ratio
    f['f'] = therm_radius/radius
    f['V0'] = V0
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
