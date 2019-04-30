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

FILETYPE='.png'

#entropy_key = 'entropy'   #'S1'
#u_key       = 'u_r'       #'u'
#V_key       = 'vorticity' #'V'
entropy_key = 'S1'
u_key       = 'u'
V_key       = 'V'
w_key       = 'w'


class DedalusIntegrator():
    """
    A simple class for post-processing thermal data
    """
    def __init__(self, nr, nz, Lr, Lz, *args, r_cheby=False, **kwargs):
        """
        Initialize the domain. Sets up a 2D (z, r) grid, with chebyshev
        decomposition in the z-direction and fourier decomposition in the r-direction.

        Inputs:
        -------
        nr, nz  : ints
            Number of coefficients in r- and z- direction
        Lr, Lz  : floats
            The size of the domain in the r- and z- direction
        *args, **kwargs : tuple, dict
            Arguments and keyword arguments for the DedalusIntegrator._polytrope_initialize() function
        """
        self.Lz, self.Lr = Lz, Lr
        if r_cheby:
            self.r_basis = de.Chebyshev('r', nr, interval=(0,Lr))
            self.z_basis = de.Fourier('z', nz, interval=(0, Lz))
            self.domain = de.Domain([self.z_basis, self.r_basis], grid_dtype=np.float64)
            self.r              = self.domain.grid(1)
            self.z              = self.domain.grid(0)
        else:
            self.r_basis = de.Fourier('r', nr, interval=(0,Lr))
            self.z_basis = de.Chebyshev('z', nz, interval=(0, Lz))
            self.domain = de.Domain([self.r_basis, self.z_basis], grid_dtype=np.float64)
            self.r              = self.domain.grid(0)
            self.z              = self.domain.grid(1)
        self.T0             = self.domain.new_field()
        self.rho0           = self.domain.new_field()
        self.rho0_z         = self.domain.new_field()
        self.grad_ln_rho0   = self.domain.new_field()
        self.fields = [self.T0, self.rho0, self.rho0_z, self.grad_ln_rho0]
        self.fd1 = self.domain.new_field()
        self.fd2 = self.domain.new_field()
        self.fd3 = self.domain.new_field()
        self.sm_fields = [self.fd1, self.fd2, self.fd3]
        self._polytrope_initialize(*args, **kwargs)

    def full_integrate(self):
        """ 
        Integrate the quantity in self.fd1 over both the r and
        z directions, then return the integrated value.
        """
        self.fd1.integrate('r',  out=self.fd2)
        self.fd2.integrate('z',  out=self.fd3)
        for f in self.sm_fields:
            f.set_scales(1, keep_data=True)
        return np.mean(self.fd3['g'])

    def field_clean(self):
        """ Clean out all values in self.fd1, self.fd2, self.fd3 """
        for f in self.sm_fields:
            f['c'] *= 0
            f.set_scales(1, keep_data=False)

    def _polytrope_initialize(self, grad_T_ad, m_ad):
        """ Fill atmosphere fields with polytropic stratification """
        self.T0['g'] = (1 + grad_T_ad*(self.z-self.Lz))
        self.T0.set_scales(1, keep_data=True)

        self.rho0['g'] = self.T0['g']**(m_ad)
        self.rho0_z['g'] = m_ad*self.T0['g']**(m_ad-1)*grad_T_ad
        self.grad_ln_rho0['g'] = -m_ad/self.T0['g']
        for f in self.fields:
            f.set_scales(1, keep_data=True)

    def calculate_streamfunc(self, w, old_r=None, old_z=None):
        """
        Calculates the streamfunction of a 2D, cylindrical, azimuthally-symmetric,
        anelastic flow. Assumes that the class-level density profile has been
        properly set.

        Parameters
        ----------
        w      : NumPy array
            Velocity data in the vertical direction
        """

        if old_r is None and old_z is None:
            w_in = w
        else:
            zi, ri = np.meshgrid(old_z, old_r)
            ro, zo = np.meshgrid(self.r.flatten(), self.z.flatten())
            w_func = RegularGridInterpolator((old_r, old_z), w, bounds_error=False, fill_value=0)
            w_in = w_func((ro, zo))

        self.rho0.set_scales(1, keep_data=True)
        self.fd1['g'] = 2*np.pi*self.r*self.rho0['g']*w_in
        self.fd1.antidifferentiate('r', ('left', 0), out=self.fd2)
        self.fd2.set_scales(1, keep_data=True)
        sf = self.fd2['g']

        if old_r is None and old_z is None:
            r = self.r.flatten()
            nz = len(self.z)
        else:
            sf_func = RegularGridInterpolator((self.z.flatten(), self.r.flatten()), sf, bounds_error=False)
            sf = sf_func((zi, ri))
            r = old_r
            nz = len(old_z)
        sf[np.isnan(sf)] = 0
        thermal_r_contour = np.zeros(nz)
        #Find psi = 0 contour
        for i in range(nz):
            if old_r is None and old_z is None:
                this_sf = sf[i,:]
            else:
                this_sf = sf[:,i]
            good = np.isfinite(this_sf)
            if np.sum(good) <= 1:
                thermal_r_contour[i] = 0
                continue
            psi_f = interp1d(r[good], this_sf[good], fill_value='extrapolate')
            if this_sf[good].min() < 0 and psi_f(r[this_sf[good].argmin()])/psi_f(r[good][-1]) < 0:
                opt = brentq(psi_f, r[this_sf[good].argmin()], r[good].max())
                thermal_r_contour[i] = opt
        return sf, thermal_r_contour


def read_file(fn, factors=(1, 1, 1), twoD=True):
    """ Reads sim info from file """
    L_factor, t_factor, s_factor = factors

    f    = h5py.File(fn, 'r')
    if twoD:
        s1  = f['tasks']['S1'].value * s_factor
        w   = f['tasks']['w'].value * L_factor/t_factor
        u   = f['tasks']['u'].value * L_factor/t_factor
        V   = f['tasks']['V'].value / t_factor
        r = f['scales']['r']['1.0'].value * L_factor 
        z = f['scales']['z']['1.0'].value * L_factor 
        rr, zz = np.meshgrid(r, z)
    else:
        s1  = f['tasks']['s y mid'].value * s_factor
        w   = f['tasks']['w y mid'].value * L_factor/t_factor
        u   = f['tasks']['w y mid'].value * L_factor/t_factor
        V   = f['tasks']['vorticity y mid'].value / t_factor
        x = f['scales']['x']['1.0'].value * L_factor 
        r = x[x >= 0]
        r_ind = (x >= 0).flatten()
        s1 = s1[:, r_ind,0,  :]
        w  =  w[:, r_ind,0,  :]
        u  =  u[:, r_ind,0,  :]
        V  =  V[:, r_ind,0,  :]
        z  = f['scales']['z']['1.0'].value * L_factor 
        zz, rr = np.meshgrid(z, r)
    f.close()
    return r, z, rr, zz, s1, w, u, V

    

def analyze_thermal_contour(root_dir, out_dir, times):
    """
    After the thermal contour is found, this analyzes the properties of the thermal in depth
    by taking integrals over the dedalus domain. Integrals are output into
    root_dir/out_dir/post_analysis.h5

    Inputs:
    -------
    root_dir    : string
        The root directory of the dedalus run, containing folders like slices/
    out_dir     : string
        The directory inside of root_dir in which output is saved
    times       : NumPy array
        An array containing the simulation time of each data point.
    """
    full_out_dir, n_rho, aspect, gamma, m_ad, Cp, Lz, Lr, grad_T_ad, g, files, prof_files, L_factor, t_factor, s_factor, twoD = get_basic_run_info(root_dir, out_dir)
    c_f = h5py.File('{:s}/fit_file.h5'.format(full_out_dir), 'r')
    w_cb = c_f['w_cb'].value
    z_cb = c_f['z_cb'].value
    c_f.close()
    c_f = h5py.File('{:s}/contour_file.h5'.format(full_out_dir), 'r')
    contours = c_f['contours'].value
    c_f.close()

    integral_fields = [ 'int_circ', 
                        'int_s1', 
                        'int_mass', 
                        'int_ke', 
                        'int_mom', 
                        'int_pe', 
                        'int_vol', 
                        'int_impulse', 
                        'int_rho_s1', 
                        'int_w', 
                        'int_circ_above', 
                        'int_rho_s1_above', 
                        'int_u', 
                        'int_area', 
                        'max_s1_therm',
                        'torus_vol',
                        'impulse_piece1',
                        'int_impulse_p1',
                        'int_impulse_p2',
                        'full_impulse',
                        'full_momentum',
                        'full_circ',
                        'full_rho_s1',
                        'full_ke',
                        'full_pe']
    integs = dict()
    for f in integral_fields:
        integs[f] = np.zeros(len(times))
    radius = np.zeros_like(times)
    height = np.zeros_like(times)
    c_f = h5py.File('{:s}/contour_file.h5'.format(full_out_dir), 'r')
    contours = c_f['contours'].value
    c_f.close()
    count = 0
    for filenum, fn in enumerate(files):
        #Read entropy, density, velocity, vorticity
        r, z, rr, zz, s1, w, u, V = read_file(fn, factors=(L_factor, t_factor, s_factor), twoD=twoD)
        f    = h5py.File(prof_files[filenum], 'r')
        time = f['scales']['sim_time'].value
        f.close()

        if filenum == 0:
            integrator = DedalusIntegrator(len(r), len(z), Lr, Lz, grad_T_ad, m_ad, r_cheby=twoD)

        for i in range(s1.shape[0]):
            count += 1
            logger.info('analyze count {}'.format(count))

            contour = contours[count-1,:]
            if contour.max() > 0:
                radius[count-1] = contour.max()
                height[count-1] = z[contour.argmax()]
            therm_mask = np.zeros_like(rr, dtype=bool)
            for k in range(len(z)):
                if twoD:
                    therm_mask[k,:] = r <= contour[k]
                else:
                    therm_mask[:,k] = r <= contour[k]

            top_indices = (z > z[contour.argmax()])*(contour/contour.max() < 1e-2)
            if True in top_indices:
                contour_top = z[top_indices][0]
            else:
                contour_top = z.max()

            integrands = {
                        'int_vol': 2*np.pi*rr, 
                        'int_area' : np.ones_like(u[i,:]), 
                        'int_circ' : V[i,:], 
                        'int_s1' : 2*np.pi*s1[i,:]*integrator.r, 
                        'int_mass' : 2*np.pi*integrator.rho0['g']*integrator.r, 
                        'int_ke' :  2*np.pi*integrator.rho0['g']*w[i,:]**2*integrator.r / 2, 
                        'int_mom' : 2*np.pi*integrator.rho0['g']*w[i,:]*integrator.r, 
                        'int_pe' : 2*np.pi*s1[i,:]*integrator.rho0['g']*g*-1*integrator.z*integrator.r,
                        'int_impulse_p1' : 2*np.pi*0.5*integrator.rho0['g']*V[i,:]*integrator.r**2,
                        'int_impulse_p2' : 2*np.pi*0.5*integrator.rho0_z['g']*u[i,:]*integrator.r**2,
                        'int_rho_s1' : 2*np.pi*integrator.rho0['g']*s1[i,:]*integrator.r, 
                        'int_rho_s1_above' : 2*np.pi*integrator.rho0['g']*s1[i,:]*integrator.r, 
                        'int_w' : 2*np.pi*w[i,:]*integrator.r, 
                        }

            integrands['int_impulse'] = integrands['int_impulse_p1'] + integrands['int_impulse_p2']
            integrands['full_impulse'] = np.copy(integrands['int_impulse'])
            integrands['full_momentum'] = np.copy(integrands['int_mom'])
            integrands['full_circ'] = np.copy(integrands['int_circ'])
            integrands['full_rho_s1'] = np.copy(integrands['int_rho_s1'])
            integrands['full_ke'] = np.copy(2*np.pi*integrator.rho0['g']*(u[i,:]**2+w[i,:]**2)/2*integrator.r)
            integrands['full_pe'] = np.copy(integrands['int_pe'])


            for name, values in integrands.items():
                if 'above' in name: 
                    integrator.fd1['g'][zz > contour_top] = values[zz > contour_top]
                elif 'full_' in name:
                    integrator.fd1['g'] = values
                else:
                    integrator.fd1['g'][therm_mask] = values[therm_mask]
                integs[name][count-1] = integrator.full_integrate()
                integrator.field_clean()

            if np.sum(therm_mask) > 0:
                integs['max_s1_therm'][count-1]  = np.max(np.abs(s1[i,:][therm_mask]))
                s1_in_therm = s1[i,:][therm_mask]
                mask = therm_mask*(s1[i,:] < 0.1*s1_in_therm.min())
                integrator.fd1['g'][mask] = rr[mask]
                integs['torus_vol'][count - 1] = integrator.full_integrate()
                integrator.field_clean()
            else:
                integs['max_s1_therm'][count-1]  = 0 

            this_w_cb  = w_cb[count-1]
            vol        = integs['int_vol'][count-1]
            this_int_w = integs['int_w'][count-1]

            circ_frac      = integs['int_circ'][count-1]/integs['full_circ'][count-1]
            circ_a_frac    = integs['int_circ_above'][count-1]/integs['full_circ'][count-1]
            entropy_frac   = integs['int_rho_s1'][count-1]/integs['full_rho_s1'][count-1]
            entropy_a_frac = integs['int_rho_s1_above'][count-1]/integs['full_rho_s1'][count-1]
            string = 'Iter: {}, w_int/vol: {:.2g},  w_cb: {:.2g}, %d: {:.2g}'.\
                    format(count, this_int_w/vol, this_w_cb, 100*(this_int_w/vol - this_w_cb)/this_w_cb)
            string += ' tot circ: {:.4e} / tot rho*s: {:.4e}'.format(integs['full_circ'][count-1], integs['full_rho_s1'][count-1])
            string += ' / circ_frac: {:.2f} / tot_entropy_frac {:.2f}'.format(circ_frac, entropy_frac)
            string += ' / circ_above/tot: {:.2f}, rhos_above/tot: {:.2f}'.format(circ_a_frac, entropy_a_frac)
            logger.info(string)

            string = 'ke: {:.4e}, pe: {:.4e}, te: {:.4e}'.format(integs['full_ke'][count-1], integs['full_pe'][count-1], integs['full_ke'][count-1]+integs['full_pe'][count-1])
            logger.info(string)


    output_file = h5py.File('{:s}/post_analysis.h5'.format(full_out_dir), 'w')
    output_file['times']       = times

    for f, fd in integs.items():
        output_file[f]    = fd
    output_file['height']      = height
    output_file['radius']      = radius
    output_file['w_cb']        = w_cb
    output_file['z_cb']        = z_cb


    output_file.close()



def calculate_contour(root_dir, out_dir, times):
    """
    Driving logic for calculating thermal volume-tracking contour. Before this function
    is run, the thermal's cb velocity must be separately fit. The volume contour is
    saved into root_dir/out_dir/contour_file.h5

    Inputs:
    -------
    root_dir    : string
        The root directory of the dedalus run, containing folders like slices/
    out_dir     : string
        The directory inside of root_dir in which output is saved
    times       : NumPy array
        An array containing the simulation time of each data point.
    """
    full_out_dir, n_rho, aspect, gamma, m_ad, Cp, Lz, Lr, grad_T_ad, g, files, prof_files, L_factor, t_factor, s_factor, twoD = get_basic_run_info(root_dir, out_dir)
    c_f = h5py.File('{:s}/fit_file.h5'.format(full_out_dir), 'r')
    w_cb = c_f['w_cb'].value
    c_f.close()

    for filenum, fn in enumerate(files):
        #Read entropy, density, velocity, vorticity
        r, z, rr, zz, s1, w, u, V = read_file(fn, factors=(L_factor, t_factor, s_factor), twoD=twoD)

        if filenum == 0:
            integrator = DedalusIntegrator(len(r), len(z), Lr, Lz, grad_T_ad, m_ad, r_cheby=True)
            count           = 0
            contours        = np.zeros((len(times), len(z)))
            radius = np.zeros_like(times)
            height = np.zeros_like(times)


        for i in range(w.shape[0]):
            count += 1
            logger.info('big calc count {}'.format(count))
           

            if not(np.isfinite(w_cb[count-1])):
                sf = np.zeros_like(w[i,:,:])
                contour = np.zeros_like(z)
            else:
                if twoD:
                    sf, contour = integrator.calculate_streamfunc(w[i,:,:] - w_cb[count-1])
                else:
                    sf, contour = integrator.calculate_streamfunc(w[i,:,:] - w_cb[count-1], old_r=r, old_z=z)
            logger.info('w_cb: {:.4e} / max contour: {:.4e}'.format(w_cb[count-1], np.max(contour)))

            if contour.max() != 0:
                #Get one index below where the contour starts coming out.
                contour_bottom = z[np.where(z[contour/contour.max() > 1e-2][0] == z)[0]-1]
            else:
                contour_bottom = 0

            if contour.max() > 0:
                top_indices = (contour/contour.max() < 1e-2) * (z > contour_bottom)
                if True in top_indices:
                    contour_top = z[top_indices][0]
                else:
                    contour_top = z.max()
            else:
                contour_top = 0

            contour[z.flatten() > contour_top]    = 0 
            contour[z.flatten() < contour_bottom] = 0
            contours[count-1,:] = contour
            if contour.max() > 0:
                radius[count-1] = contour.max()
                height[count-1] = z[contour.argmax()]

            output_file = h5py.File('{:s}/contour_file.h5'.format(full_out_dir), 'w')
            output_file['z']                = z
            output_file['contours']         = contours
            output_file.close()


def differentiate(x, y): 
    """ Numerical differentiation. Output has 4 fewer points than input. """
    dx = np.mean(x[1:] - x[:-1])
    return (-y[4:] + 8*y[3:-1] - 8*y[1:-3] + y[:-4]) / (12*dx)


def fit_w_cb(root_dir, out_dir, times, iterative=False):
    """
    Fit thermal bulk velocity (cloud bottom, 'cb') according to the
    developed theory of stratified thermal evolution.

    Output information from the fit is stored in root_dir/out_dir/fit_file.h5

    Inputs:
    -------
    root_dir    : string
        The root directory of the dedalus run, containing folders like slices/
    out_dir     : string
        The directory inside of root_dir in which output is saved
    times       : NumPy array
        An array containing the simulation time of each data point.
    iterative   : bool, optional
        If True, use output from a previous fit to get this fit.
    """
    full_out_dir, n_rho, aspect, gamma, m_ad, Cp, Lz, Lr, grad_T_ad, g, files, prof_files, L_factor, t_factor, s_factor, twoD = get_basic_run_info(root_dir, out_dir)

    c_f = h5py.File('{:s}/z_cb_file.h5'.format(full_out_dir), 'r')
    z_cb = c_f['z_cb'].value
    z = c_f['z'].value
    vortex_height = c_f['vortex_height'].value
    vortex_radius = c_f['vortex_radius'].value
    vortex_rho    = c_f['vortex_rho'].value
    vortex_T      = c_f['vortex_T'].value
    R             = c_f['R'].value
    Rz            = c_f['Rz'].value
    frac          = c_f['frac'].value
    A             = c_f['A'].value
    circ          = c_f['circ'].value
    B             = c_f['B'].value
    momentum      = c_f['momentum'].value
    c_f.close()


    fit_t = get_good_times(z_cb, Lz, L_max=0.7, L_min=0.15)

    #Estimates
    B_est = np.mean(B[fit_t])
    circ_est = np.mean(circ[fit_t])
    f_est    = np.mean(frac[fit_t])
    A_est    = np.mean(A[fit_t])
    V0_est   = 4*np.pi/3/A_est

    circ0 = circ[fit_t][0]
    I    = vortex_T[fit_t]**m_ad * vortex_radius[fit_t]**2 * circ_est/2
    (a_i, i0), pcov = scop.curve_fit(linear_fit, times[fit_t], I)
    (a_mom, m0), pcov = scop.curve_fit(linear_fit, times[fit_t], momentum[fit_t])


    logger.info('fit i0: {:.4e} // fit m0: {:.4e} // gamma: {:.4e}'.format(i0, m0, circ_est))

    #Set ranges for parameters in solver
    if iterative:
        output_file = h5py.File('{:s}/iterative_file.h5'.format(full_out_dir), 'r')
        B = output_file['B']        .value
        f = output_file['f']        .value
        V0 = output_file['V0']      .value
        M0 = output_file['M0']      .value
        I0 = output_file['I0']      .value
        Gamma = output_file['Gamma'].value
        output_file.close()
        beta_min, beta_max = 0.5, 1
        T0_min, T0_max = 0, vortex_T[0]
        bounds = ((beta_min, T0_min), 
                  (beta_max, T0_max))
        p = (0.5, T0_max/2)
        this_T_theory     = lambda times, beta, T0:     theory_T(times, B, M0, I0, Gamma, f, V0, T0, beta, grad_T_ad=grad_T_ad, m_ad=m_ad)
        this_dT_dt_theory = lambda times, beta, T0: theory_dT_dt(times, B, M0, I0, Gamma, f, V0, T0, beta, grad_T_ad=grad_T_ad, m_ad=m_ad)
        (beta, T0), pcov = scop.curve_fit(this_T_theory, times[fit_t], vortex_T[fit_t], bounds=bounds, p0=p)
        cb_T_fit =  this_T_theory(times, beta, T0)
        dT_dt = this_dT_dt_theory(times, beta, T0)
    else:
        B_min, B_max = B_est, 0.5*B_est
        Gamma_min, Gamma_max = circ_est, 0.6*circ_est
        M_min, M_max = 5*m0, -0.2*m0
        if i0 < 0:
            I_min, I_max = 5*i0, -0.2*i0
        else:
            I_min, I_max = -0.2*i0, 5*i0
        beta_min, beta_max = 0.5, 0.55#2./3 
        V0_min, V0_max = 0.99*V0_est, 1*V0_est
        f_min, f_max   = 0.5*f_est, 1.5*f_est
        T0_min, T0_max = 0, 1*vortex_T[0]#, 1.5*vortex_T[0]
        bounds = ((B_min, M_min, I_min, Gamma_min, beta_min, T0_min, f_min), 
                  (B_max, M_max, I_max, Gamma_max, beta_max, T0_max, f_max))

        p = (B_min, m0, i0, Gamma_min, beta_min, vortex_T[0], f_est)

        #Wrap theory functions with assumptions and atmospheric info
        this_T_theory     = lambda times, B, M0, I0, Gamma, beta, T0, f:     theory_T(times, B, M0, I0, Gamma, f, V0_est, T0, beta, grad_T_ad=grad_T_ad, m_ad=m_ad)
        this_dT_dt_theory = lambda times, B, M0, I0, Gamma, beta, T0, f: theory_dT_dt(times, B, M0, I0, Gamma, f, V0_est, T0, beta, grad_T_ad=grad_T_ad, m_ad=m_ad)
        (B, M0, I0, Gamma, beta, T0, f), pcov = scop.curve_fit(this_T_theory, times[fit_t], vortex_T[fit_t], bounds=bounds, p0=p, maxfev=int(1e4))
        cb_T_fit = this_T_theory(times, B, M0, I0, Gamma, beta,  T0, f)
        dT_dt = this_dT_dt_theory(times, B, M0, I0, Gamma, beta, T0, f)
        V0 = V0_est

    vortex_dT_dt = differentiate(times, vortex_T)#np.diff(vortex_T)/np.diff(times)
    w_cb = dT_dt/grad_T_ad
    if np.isnan(w_cb[0]) or np.isinf(w_cb[0]):
        w_cb[0] = 0

    #Output info & safe to file
    logger.info('theory B {:.4e}, M0 {:.4e}, I0 {:.4e}, beta {:.4e}, f {:.4e}, V0 {:.4e}, Gamma {:.4e}, T0 {:.2f}'.format(B, M0, I0, beta, f, V0, Gamma, T0))
    logger.info('w_cb {}'.format(w_cb[fit_t]))
    logger.info('vortex dT_dt/gradT {}'.format(vortex_dT_dt[fit_t[2:-2]]/grad_T_ad))
    logger.info('theory T  {}'.format(cb_T_fit[fit_t]))
    logger.info('vortex T  {}'.format(vortex_T[fit_t]))
    logger.info('(vortex - theory)/vortex T {}'.format(np.mean(np.abs((1 - cb_T_fit/vortex_T)[fit_t]))))
    logger.info('(vortex - theory)/vortex dT/dz {}'.format(np.mean(np.abs((1 - dT_dt[2:-2]/vortex_dT_dt)[fit_t[2:-2]]))))

    output_file = h5py.File('{:s}/fit_file.h5'.format(full_out_dir), 'w')
    output_file['w_cb']             = w_cb
    output_file['z_cb']             = z_cb
    output_file['cb_T_fit']         = cb_T_fit
    output_file['vortex_height']    = vortex_height
    output_file['vortex_radius']    = vortex_radius
    output_file['vortex_rho']       = vortex_rho
    output_file['vortex_T']         = vortex_T
    output_file['fit_B']            = B
    output_file['fit_M0']           = M0
    output_file['fit_I0']           = I0
    output_file['fit_beta']         = beta
    output_file['fit_Gamma']        = Gamma
    output_file['fit_f']            = f
    output_file['fit_V0']           = V0
    output_file['fit_T0']           = vortex_T[0]
    output_file.close()


def get_basic_run_info(root_dir, out_dir):
    """
    Inputs:
    -------
    root_dir    : string
        The root directory of the dedalus run, containing folders like slices/
    out_dir     : string
        The directory inside of root_dir in which output is saved
    """
    n_rho = float(root_dir.split('_nrho')[-1].split('_')[0])
    aspect = float(root_dir.split('_aspect')[-1].split('_')[0].split('/')[0])
    gamma = 5./3
    m_ad = 1/(gamma-1)
    Cp = gamma*m_ad
    Lz_true = np.exp(n_rho/m_ad) - 1

    if '_2D' in root_dir:
        twoD = True
        Lz = 20
        Lr = aspect*Lz
    else:
        twoD = False
        epsilon = float(root_dir.split('_eps')[-1].split('_')[0].split('/')[0])
        Lz = 20#Lz_true
        Lr = (Lz/20)*aspect/2

    grad_T_ad = -Lz_true/Lz
    g = -grad_T_ad * (1 + m_ad)

 
    full_out_dir = '{:s}/{:s}/'.format(root_dir, out_dir)
    if not os.path.exists('{:s}/'.format(full_out_dir)):
        os.mkdir('{:s}'.format(full_out_dir))


    #Read in times
    files = glob.glob("{:s}/slices/slices_s*.h5".format(root_dir))
    prof_files = glob.glob("{:s}/profiles/profiles*.h5".format(root_dir))
    nf  = [int(f.split('.h5')[0].split('_s')[-1]) for f in files]
    npr = [int(f.split('.h5')[0].split('_s')[-1]) for f in prof_files]
    n, files = zip(*sorted(zip(nf, files)))
    n, prof_files = zip(*sorted(zip(npr, prof_files)))

    if twoD:
        t_factor = s_factor = L_factor = 1
    else:
        t_b_true = np.sqrt((Lz_true)/(epsilon))
        t_b = np.sqrt(Lz) 

        t_factor = t_b/t_b_true
        s_factor = Cp/epsilon
        L_factor = -1/grad_T_ad
        
    return full_out_dir, n_rho, aspect, gamma, m_ad, Cp, Lz, Lr, grad_T_ad, g, files, prof_files, L_factor, t_factor, s_factor, twoD 


def get_good_times(z, Lz, L_max=0.7, L_min=0.15):
        """
        A simple function for creating a boolean indexing array. When the thermal is
        below L_max * Lz and above L_min * Lz, the result is True. Otherwise, it's false.
        This function also accounts for the ability of a thermal to go through the bottom,
        and so it only counts the "first fall" in a 2D sim.

        Inputs:
        -------
        z   : NumPy array
            The z-value of the thermal at each point in time
        Lz  : float
            The depth of the atmosphere
        L_max   : float, optional
            The fraction of Lz above which the thermal shouldn't be tracked
        L_min   : float, optional
            The fraction of Lz below which the thermal shouldn't be tracked

        Outputs:
        -------
        fit_t   : NumPy array, (dtype: bool)
            A conditional indexing array that is True when the thermal is within the allowed range

        """
        #Next, fit cb height and velocity
        fit_t = (z < L_max*Lz)*(z > L_min*Lz)

        #Ensure that we only track the first fall of the thermal (in case it goes through the fourier bottom of the domain)
        found_therm = False
        therm_done  = False
        for i in range(len(z)):
            if fit_t[i] and therm_done:
                fit_t[i] = False
            if fit_t[i] and not found_therm:
                found_therm = True
            if found_therm and not fit_t[i] and not therm_done:
                therm_done=True
        fit_t[0] = False
        return fit_t




def linear_fit(x, a, b):
    """ A linear fit: y = a* x + b """
    return a*x + b

def measure_cb(root_dir, out_dir, times):
    """
    Function that measures some basic information about the thermal. The following quantities
    are output into root_dir/thermal_analysis/z_cb_file.h5:
         times         : simulation times
         z             : simulation grid in vertical direction
         z_cb          : cloud-bottom height of the thermal at each time
         vortex_height : height of the thermal's vortex ring
         vortex_radius : radius from axis of symmetry of the thermal's vortex ring
         vortex_rho    : atmospheric density at height of thermal vortex ring
         vortex_T      : atmospheric temperature at height of thermal vortex ring
         circ          : The circulation in the full domain
         B             : The total volume-integrated buoyancy in the full domain
         momentum      : The total volume-integrated vertical momentum in the full domain
         R             : The radius of the full thermal.
         Rz            : The radius of the thermal in the vertical direction.
         frac          : vortex_radius/R
         A             : R/Rz


    Inputs:
    -------
    root_dir    : string
        The root directory of the dedalus run, containing folders like slices/
    out_dir     : string
        The directory inside of root_dir in which output is saved
    times       : NumPy array
        An array containing the simulation time of each data point.

    """
    full_out_dir, n_rho, aspect, gamma, m_ad, Cp, Lz, Lr, grad_T_ad, g, files, prof_files, L_factor, t_factor, s_factor, twoD = get_basic_run_info(root_dir, out_dir)
    #Make empty space to track important quantities
    vortex_height    = np.zeros(len(times))
    vortex_radius    = np.zeros(len(times))
    vortex_T         = np.zeros(len(times))
    vortex_rho       = np.zeros(len(times))
    z_cb             = np.zeros(len(times))

    circ             = np.zeros(len(times))
    B                = np.zeros(len(times))
    momentum         = np.zeros(len(times))


    R                = np.zeros(len(times)) #horizontal radius
    Rz               = np.zeros(len(times)) #vertical radius
    frac             = np.zeros(len(times)) # = vortex_radius/R
    A                = np.zeros(len(times)) # = R / Rz
    if os.path.exists('{:s}/z_cb_file.h5'.format(full_out_dir)):
        return
    else:
        count = 0
        #First, get some basic info about the thermal
        for filenum, fn in enumerate(files):
            #Read entropy, density, velocity, vorticity
            r, z, rr, zz, s1, w, u, V = read_file(fn, factors=(L_factor, t_factor, s_factor), twoD=twoD)

            if filenum == 0:
                integrator = DedalusIntegrator(len(r), len(z), Lr, Lz, grad_T_ad, m_ad, r_cheby=twoD)

            for i in range(s1.shape[0]):
                count += 1
                #Circulation
                print(V[i,:,:].shape)
                integrator.fd1['g'] = V[i,:,:]
                circ[count-1] = integrator.full_integrate()
                integrator.field_clean()
                #Buoyancy
                integrator.fd1['g'] = 2*np.pi*integrator.rho0['g']*s1[i,:,:]*integrator.r
                B[count-1] = integrator.full_integrate()
                integrator.field_clean()
                #momentum
                integrator.fd1['g'] = 2*np.pi*integrator.rho0['g']*w[i,:,:]*integrator.r
                momentum[count-1] = integrator.full_integrate()
                integrator.field_clean()

                integrator.fd1['g'] = integrator.rho0['g']*s1[i,:,:]*integrator.r
                integrator.fd1.integrate('r', out=integrator.fd2)
                integrator.fd1.set_scales(1, keep_data=True)
                integrator.fd1['g'] = integrator.rho0['g']*s1[i,:,:]
                integrator.fd1.integrate('z', out=integrator.fd3)
                integrator.fd2.set_scales(1, keep_data=True)
                integrator.fd3.set_scales(1, keep_data=True)
                if twoD:
                    s_prof   = np.copy(integrator.fd2['g'][:,0])
                    s_prof_r = np.copy(integrator.fd3['g'][0,:])
                else:
                    s_prof   = np.copy(integrator.fd2['g'][0,:])
                    s_prof_r = np.copy(integrator.fd3['g'][:,0])

                therm_r = integrator.r.flatten()[s_prof_r < 0.2*s_prof_r.min()]
                therm_z = integrator.z.flatten()[s_prof   < 0.2*s_prof.min()]

                #Find thermal location, roughly
                if twoD:
                    s_prof_dz = integrator.fd2.differentiate('z')['g'][:,0]
                else:
                    s_prof_dz = integrator.fd2.differentiate('z')['g'][0,:]
                z_f = interp1d(integrator.z.flatten(), s_prof_dz, fill_value='extrapolate')
                try:
                    vortex_height[count - 1] = brentq(z_f, therm_z[0], therm_z[-1])
                except:
                    vortex_height[count - 1] = integrator.z.flatten()[s_prof.argmin()]

                if twoD:
                    s_prof_r_dr = integrator.fd3.differentiate('r')['g'][0,:]
                else:
                    s_prof_r_dr = integrator.fd3.differentiate('r')['g'][:,0]
#                profile_r = s_prof_r - 0.2*s_prof_r.min()
                r_f = interp1d(integrator.r.flatten(), s_prof_r_dr, fill_value='extrapolate')
                guess = integrator.r.flatten()[s_prof_r.argmin()]

                try:
                    if guess < Lr/20:
                        upper = Lr/2
                    else:
                        upper = guess*1.2
                    lower = guess*0.8
                    vortex_radius[count - 1] = brentq(r_f, lower, upper)
                except:
                    vortex_radius[count - 1] = 0

                profile_z = s_prof - 0.1*s_prof.min()
                z_f2 = interp1d(integrator.z.flatten(), profile_z, fill_value='extrapolate')
                try:
                    this_z_cb = brentq(z_f2, 0, vortex_height[count-1])
                except:
                    this_z_cb = integrator.z.flatten()[profile_z < profile_z.min()*0.1][0]

                therm_bot, therm_top, this_Rz = therm_z[0], therm_z[-1], (therm_z[-1]-therm_z[0])/2
                this_R = therm_r[-1]
                this_f = vortex_radius[count - 1] / this_R
                this_A = this_R / this_Rz

                logger.info('Rz: {:.2f}, R: {:.2f}, vortex_radius: {:.2f}, A: {:.2f}, f: {:.2f}'.format(this_Rz, this_R, vortex_radius[count - 1], this_A, this_f))

                vortex_rho[count - 1]    = np.mean(integrator.rho0.interpolate(z=vortex_height[count - 1])['g'])
                vortex_T[count - 1]      = np.mean(integrator.T0.interpolate(z=vortex_height[count - 1])['g'])

                R[count - 1]      = this_R
                Rz[count - 1]     = this_Rz
                frac[count - 1]   = this_f
                A[count - 1]      = this_A

                z_cb[count - 1] = this_z_cb
                logger.info('cb calc count {}, t = {:.2e} / z_cb {:.2e} / (r,z) = ({:.2e}, {:.2e})'.format(count, times[count-1], this_z_cb, vortex_radius[count-1], vortex_height[count-1]))
        output_file = h5py.File('{:s}/z_cb_file.h5'.format(full_out_dir), 'w')
        output_file['times']            = times
        output_file['z']                 = z
        output_file['z_cb']             = z_cb
        output_file['vortex_height']    = vortex_height
        output_file['vortex_radius']    = vortex_radius
        output_file['vortex_rho']       = vortex_rho
        output_file['vortex_T']         = vortex_T
        output_file['circ']             = circ
        output_file['B']                = B
        output_file['momentum']         = momentum
        output_file['R']                = R
        output_file['Rz']               = Rz
        output_file['frac']             = frac
        output_file['A']                = A
        output_file.close()




def plot_and_fit_trace(x, y, fit_ind=None, fit_func=linear_fit, 
                        fig_name='fig', labels=['x','y'], fit_str_format=None, **kwargs):
    """
    Fit a time series, then plot and save a figure of the data and the fit.

    Parameters
    ----------
    x, y    : NumPy arrays, 1-dimensional
        The x- and y- data
    fit_ind : NumPy array of bools, optional
        If not None, the array of indices to use while fitting
    fit_func : Python function, optional
        The function to use while fitting with scipy
    fig_name : string, optional
        The name of the figure. Filetype is .png
    fit_str_format : string, optional
        A formatting string in which to put fit parameters, if not None
    labels   : list, optional
        The x- and y- labels
    kwargs   : dict, optional
        keyword arguments for the scipy.optimize.curve_fit function

    Returns
    -------
    fig      : matplotlib Figure object
        The figure on which the data and fit are plotted
    (ax1, ax2) : tuple of matplotl Axes objects
        ax1 is the figure on which data & fit are plotted. ax2 is the goodness of fit.
    best_fit   : a NumPy array
        An array containing the optimzied fit
    fits       : tuple
        A tuple containing the parameters used with fit_func() to make the fit.
    """
    if fit_ind is None:
        fits, pcov = scop.curve_fit(fit_func, x, y, **kwargs)
    else:
        fits, pcov = scop.curve_fit(fit_func, x[fit_ind], y[fit_ind], **kwargs)
    best_fit  = fit_func(x, *fits)

        
    if fit_str_format is not None:
        label_str = fit_str_format.format(*fits)


    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(2,1,1)
    plt.plot(x, y)
    if fit_str_format is not None:
        plt.plot(x, best_fit, '--', label = r'{:s}'.format(label_str))
        plt.legend(loc='best')
    else:
        plt.plot(x, best_fit, '--')

    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.ylabel(labels[1])

    ax2 = fig.add_subplot(2,1,2)
    plt.axhline(1, dashes=(5,1.5), c='k')
    plt.plot(x, y/best_fit)
    plt.xlim(x.min(), x.max())
    plt.ylim(0.9, 1.1)
    plt.ylabel(r'{:s} / fit'.format(labels[1]))
    print(labels)
    plt.xlabel(labels[0])

    return fig, (ax1, ax2), best_fit, fits



def plot_colormeshes(root_dir, out_dir, times):
    """
    Plot colormeshes of the thermal evolution.  The azimuthally averaged values of
    entropy, velocity (radial and vertical), and vorticity are plotted. The outline
    of the thermal, as found by the thermal tracker, is overplotted if available.

    Inputs:
    -------
    root_dir    : string
        The root directory of the dedalus run, containing folders like slices/
    out_dir     : string
        The directory inside of root_dir in which output is saved
    times       : NumPy array
        An array containing the simulation time of each data point.
    """

    full_out_dir, n_rho, aspect, gamma, m_ad, Cp, Lz, Lr, grad_T_ad, g, files, prof_files, L_factor, t_factor, s_factor, twoD = get_basic_run_info(root_dir, out_dir)
    if os.path.exists('{:s}/fit_file.h5'.format(full_out_dir)):
        c_f = h5py.File('{:s}/fit_file.h5'.format(full_out_dir), 'r')
        z_cb = c_f['z_cb'].value
        radius = c_f['vortex_radius'].value
        height = c_f['vortex_height'].value
        c_f.close()
    else:
        radius, height, z_cb = np.zeros_like(times), np.zeros_like(times), np.zeros_like(times)
    if os.path.exists('{:s}/contour_file.h5'.format(full_out_dir)):
        c_f = h5py.File('{:s}/contour_file.h5'.format(full_out_dir), 'r')
        contours = c_f['contours'].value
        c_f.close()
    else:
        contours = None 


    fig = plt.figure(figsize=(7, 6))
    ax1 = fig.add_subplot(1,4,1)
    ax2 = fig.add_subplot(1,4,2)
    ax3 = fig.add_subplot(1,4,3)
    ax4 = fig.add_subplot(1,4,4)
    axs  = [ax1, ax2, ax3, ax4]
    caxs = []
    for i, ax in enumerate(axs):
        divider = make_axes_locatable(ax)
        caxs.append(divider.new_vertical(size="5%", pad=0.3, pack_start=False))
        fig.add_axes(caxs[-1])

    count = 0

    for fn, fp in zip(files, prof_files):
        #Read entropy, density, velocity, vorticity
        r, z, rr, zz, s1, w, u, V = read_file(fn, factors=(L_factor, t_factor, s_factor), twoD=twoD)
        f    = h5py.File(fp, 'r')
        time = f['scales']['sim_time'].value
        f.close()
        
        if contours is None:
            contours = np.zeros((len(times),len(z)))


        #Loop over writes & plot colormeshes
        for i in range(s1.shape[0]):
            count += 1
            logger.info('plotting count {}'.format(count))
            contour = contours[count-1,:]
            #Only plot the contour around the thermal (sometimes it picks up detrained tail if you don't do this)
            if contour.max() != 0:
                contour_bottom = z[np.where(z[contour/contour.max() > 1e-2][0] == z)[0]-1]
            else:
                contour_bottom = 0
            top_indices = (contour/contour.max() < 1e-2) * (z > contour_bottom)
            if True in top_indices:
                contour_top = z[top_indices][0]
            else:
                contour_top = z.max()
            good_contour = (z >= contour_bottom)*(z <= contour_top)

            for j, info in enumerate(zip(axs, caxs, [s1[i,:,:], w[i,:,:], u[i,:,:], V[i,:,:]], ['s1', 'w', 'u', r'$\omega$'])):
                ax, cax, fd, title = info
                maxv = np.abs(fd).max()
                cmesh = ax.pcolormesh(rr, zz, fd, cmap='RdBu_r', vmin=-maxv, vmax=maxv)
                cbar = fig.colorbar(cmesh, cax=cax, orientation="horizontal", ticks=[-maxv*.75,0,maxv*.75])
                cax.set_xticklabels(['', r'$\pm{:.2e}$'.format(maxv), ''])
                cax.xaxis.set_ticks_position('none')
                cax.set_title(title)
                ax.set_xlim(0, 0.25*Lz)
                ax.set_ylim(z.min(), z.max())
                ax.axhline(z_cb[count-1], lw=0.5, c='k', ls='--')
                ax.scatter(radius[count-1], height[count-1], color='black', s=2)
                ax.plot(contour[good_contour], z[good_contour], c='k', lw=0.5)
                ax.axhline(contour_top, c='grey', lw=0.5)
                if j > 0: ax.set_yticks([])

            plt.suptitle('sim_time = {:.4e}'.format(times[count-1]))
            fig.savefig('{:s}/radial_fields_{:04d}.png'.format(full_out_dir,count), dpi=150, figsize=(10,6))
            [ax.cla() for ax in axs]
            [cax.cla() for cax in caxs]



def theory_C(B, Gamma, f, V0, grad_T_ad=-1):
    """ The constant in stratified thermals theory """
    return np.pi**(3./2) * f**3 * Gamma * grad_T_ad / (V0 * B)

def theory_T(t, B, M0, I0, Gamma, f,  V0, T0, beta, grad_T_ad=-1, m_ad=1.5):
    """ The temperature profile (vs time) from stratified thermals theory """
    C0 = theory_C(B, Gamma, f, V0, grad_T_ad=grad_T_ad)
    tau = (B * t + I0)/Gamma
    tau0 = I0/Gamma
    Tpow = 1 - (m_ad/2)
    tau_int  = np.sqrt(tau )*(Gamma*beta - (M0 - beta*I0)/tau)
    tau_int0 = np.sqrt(tau0)*(Gamma*beta - (M0 - beta*I0)/tau0)
    return (2*Tpow*C0*(tau_int-tau_int0) + T0**(Tpow))**(1./Tpow)

def theory_dT_dt(t, B, M0, I0, Gamma, f,  V0, T0, beta, grad_T_ad=-1, m_ad=1.5):
    """ The temperature derivative (vs time) from stratified thermals theory """
    C0 = theory_C(B, Gamma, f, V0, grad_T_ad=grad_T_ad)
    this_T = theory_T(t, B, M0, I0, Gamma, f, V0, T0, beta, grad_T_ad=grad_T_ad, m_ad=m_ad)
    tau = (B * t + I0)/Gamma
    dtau_dt = B/Gamma
    return C0*dtau_dt*this_T**(m_ad/2)*(beta*Gamma/np.sqrt(tau) + (M0 - beta*I0)/np.sqrt(tau**3))




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
        analyze_thermal_contour(root_dir, out_dir, times)

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
    fit_f   = output_file['fit_f']           .value 
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

    #Eqn3, r = f R, vortex core radius / total radius is a fraction and constant
    logger.info('plotting eqn3, r = f R')
    f = therm_radius/radius
    fig, axs, f_fit, (df_dt, f_Fit) = plot_and_fit_trace(scale_t[good*np.isfinite(f)], f[good*np.isfinite(f)], fit_ind=fit_t[good*np.isfinite(f)], fit_func=linear_fit, 
                       labels=['t',r'$f = r/R$'],
                       fit_str_format='{:.2g} $t$ + {:.2g}')
    axs[0].axhline(fit_f, c='green', dashes=(2,1))
    fig.savefig('{:s}/rR_fraction_v_time{:s}'.format(full_out_dir, FILETYPE), dpi=200, bbox_inches='tight')
    plt.close(fig)

    #Eqn4, volume ~ R^3, or r^3
    logger.info('plotting eqn4, V/R^3 = V_0')
    V0 = volumes/radius**3
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
    V_est = V0/(f_Fit+df_dt*scale_t)**3 * r_fit**3
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
    f['f']  = np.mean(f_fit)
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
