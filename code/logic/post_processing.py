"""
Analysis functionality for a 2D, cylindrically symmetric thermal's properties 

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

def differentiate(x, y): 
    """ Numerical differentiation. Output has 4 fewer points than input. """
    dx = np.mean(x[1:] - x[:-1])
    return (-y[4:] + 8*y[3:-1] - 8*y[1:-3] + y[:-4]) / (12*dx)

def linear_fit(x, a, b):
    """ A linear fit: y = a* x + b """
    return a*x + b

def theory_C(B0, Gamma0, f, chi, beta, grad_T_ad=-1):
    """ The constant in stratified thermals theory """
    return (beta * grad_T_ad * Gamma0 * np.pi**(3./2) / f) * np.sqrt(Gamma0/(B0 * chi))

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

class DedalusIntegrator():
    """
    A simple class for post-processing thermal data
    """
    def __init__(self, nr, nz, Lr, Lz, *args, r_cheby=False, twoD=False, **kwargs):
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
            bounds = (0, Lz)
            self.r_basis = de.Chebyshev('r', nr, interval=(0,Lr))
            self.z_basis = de.Fourier('z', nz, interval=bounds)
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


class ThermalPostProcessor():

    def __init__(self, root_dir, out_dir):
        """
        Initializ the post-processor.
        Inputs:
        -------
        root_dir    : string
            The root directory of the dedalus run, containing folders like slices/
        out_dir     : string
            The directory inside of root_dir in which output is saved
        """
        self.n_rho = float(root_dir.split('_nrho')[-1].split('_')[0])
        self.aspect = float(root_dir.split('_aspect')[-1].split('_')[0].split('/')[0])
        self.gamma = 5./3
        self.m_ad = 1/(self.gamma-1)
        self.Cp = self.gamma*self.m_ad
        self.Lz_true = np.exp(self.n_rho/self.m_ad) - 1

        if '_2D' in root_dir:
            self.twoD = True
            self.epsilon = 0
            self.Lz = 20
            self.Lr = self.aspect*self.Lz
        else:
            self.twoD    = False
            self.epsilon = float(root_dir.split('_eps')[-1].split('_')[0].split('/')[0])
            self.Lz = 20
            self.Lr = self.aspect*self.Lz/2

        self.grad_T_ad = -self.Lz_true/self.Lz
        self.g = -self.grad_T_ad * (1 + self.m_ad)

     
        self.full_out_dir = '{:s}/{:s}/'.format(root_dir, out_dir)
        if not os.path.exists('{:s}/'.format(self.full_out_dir)):
            os.mkdir('{:s}'.format(self.full_out_dir))


        #Read in times
        files = glob.glob("{:s}/slices/slices_s*.h5".format(root_dir))
        prof_files = glob.glob("{:s}/profiles/profiles*.h5".format(root_dir))
        nf  = [int(f.split('.h5')[0].split('_s')[-1]) for f in files]
        npr = [int(f.split('.h5')[0].split('_s')[-1]) for f in prof_files]
        n, self.files = zip(*sorted(zip(nf, files)))
        n, self.prof_files = zip(*sorted(zip(npr, prof_files)))

        if self.twoD:
            self.t_factor = self.s_factor = self.L_factor = 1
        else:
            self.t_b_true = np.sqrt((self.Lz_true)/(self.epsilon))
            self.t_b = np.sqrt(self.Lz) 

            self.t_factor = self.t_b/self.t_b_true
            self.s_factor = self.Cp/self.epsilon
            self.L_factor = -1/self.grad_T_ad

        count           = 0
        self.times           = []
        for fn in self.files:
            logger.info('reading time on file {:s}'.format(fn))
            f = h5py.File(fn, 'r')
            these_times = f['scales']['sim_time'].value
            [self.times.append(t) for t in these_times]
        self.times = np.array(self.times)*self.t_factor
        self.times -= self.times[0]



    def read_file(self, fn):
        """ Reads sim info from file """
        f    = h5py.File(fn, 'r')
        if self.twoD:
            s1  = f['tasks']['S1'].value * self.s_factor
            w   = f['tasks']['w'].value  * self.L_factor/self.t_factor
            u   = f['tasks']['u'].value  * self.L_factor/self.t_factor
            V   = f['tasks']['V'].value  / self.t_factor
            r   = f['scales']['r']['1.0'].value * self.L_factor 
            z   = f['scales']['z']['1.0'].value * self.L_factor 
            rr, zz = np.meshgrid(r, z)
        else:
            s1  = f['tasks']['s y mid'].value * self.s_factor
            w   = f['tasks']['w y mid'].value * self.L_factor/self.t_factor
            u   = f['tasks']['u y mid'].value * self.L_factor/self.t_factor
            V   = f['tasks']['vorticity y mid'].value / self.t_factor
            x = f['scales']['x']['1.0'].value * self.L_factor 
            r = x[x >= 0]
            r_ind = (x >= 0).flatten()
            s1 = s1[:, r_ind,0,  :]
            w  =  w[:, r_ind,0,  :]
            u  =  u[:, r_ind,0,  :]
            V  =  V[:, r_ind,0,  :]
            z  = f['scales']['z']['1.0'].value * self.L_factor 
            zz, rr = np.meshgrid(z, r)
        f.close()
        return r, z, rr, zz, s1, w, u, V


    def calculate_contour(self, iterative=False):
        """
        Driving logic for calculating thermal volume-tracking contour. Before this function
        is run, the thermal's cb velocity must be separately fit. The volume contour is
        saved into root_dir/out_dir/contour_file.h5
        """
        if iterative:
            c_f = h5py.File('{:s}/fit_file.h5'.format(self.full_out_dir), 'r')
            w_cb = c_f['fit_w'].value
            c_f.close()
        else:
            f = h5py.File('{:s}/z_cb_file.h5'.format(self.full_out_dir), 'r')
            w_cb = np.zeros_like(self.times)
            vortex_w = f['vortex_w'].value
            w_cb[2:-2] = vortex_w
            f.close()

        for filenum, fn in enumerate(self.files):
            #Read entropy, density, velocity, vorticity
            r, z, rr, zz, s1, w, u, V = self.read_file(fn)

            if filenum == 0:
                integrator = DedalusIntegrator(len(r), len(z), self.Lr, self.Lz, self.grad_T_ad, self.m_ad, r_cheby=True, twoD=self.twoD)
                count           = 0
                contours        = np.zeros((len(self.times), len(z)))
                radius = np.zeros_like(self.times)
                height = np.zeros_like(self.times)


            for i in range(w.shape[0]):
                count += 1
                logger.info('big calc count {}'.format(count))
               

                if not(np.isfinite(w_cb[count-1])) or (not iterative)*(count < 3 or count > len(self.times) -2):
                    sf = np.zeros_like(w[i,:,:])
                    contour = np.zeros_like(z)
                else:
                    if self.twoD:
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

                output_file = h5py.File('{:s}/contour_file.h5'.format(self.full_out_dir), 'w')
                output_file['z']                = z
                output_file['contours']         = contours
                output_file.close()

    def get_good_times(self, z, L_max=0.7, L_min=0.15):
            """
            A simple function for creating a boolean indexing array. When the thermal is
            below L_max * Lz and above L_min * Lz, the result is True. Otherwise, it's false.
            This function also accounts for the ability of a thermal to go through the bottom,
            and so it only counts the "first fall" in a 2D sim.

            Inputs:
            -------
            z   : NumPy array
                The z-value of the thermal at each point in time
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
            fit_t = (z < L_max*self.Lz)*(z > L_min*self.Lz)

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

    def measure_cb(self):
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
        """
        #Make empty space to track important quantities
        vortex_height    = np.zeros(len(self.times))
        vortex_radius    = np.zeros(len(self.times))
        vortex_T         = np.zeros(len(self.times))
        vortex_rho       = np.zeros(len(self.times))
        z_cb             = np.zeros(len(self.times))

        circ             = np.zeros(len(self.times))
        B                = np.zeros(len(self.times))
        momentum         = np.zeros(len(self.times))


        R                = np.zeros(len(self.times)) #horizontal radius
        Rz               = np.zeros(len(self.times)) #vertical radius
        frac             = np.zeros(len(self.times)) # = vortex_radius/R
        A                = np.zeros(len(self.times)) # = R / Rz
        if os.path.exists('{:s}/z_cb_file.h5'.format(self.full_out_dir)):
            return
        else:
            count = 0
            #First, get some basic info about the thermal
            for filenum, fn in enumerate(self.files):
                #Read entropy, density, velocity, vorticity
                r, z, rr, zz, s1, w, u, V = self.read_file(fn)

                if filenum == 0:
                    integrator = DedalusIntegrator(len(r), len(z), self.Lr, self.Lz, self.grad_T_ad, self.m_ad, r_cheby=self.twoD, twoD=self.twoD)

                for i in range(s1.shape[0]):
                    count += 1
                    integrator.fd1['g'] = integrator.rho0['g']*s1[i,:,:]*integrator.r
                    integrator.fd1.integrate('r', out=integrator.fd2)
                    integrator.fd1.set_scales(1, keep_data=True)
                    integrator.fd2.set_scales(1, keep_data=True)
                    if self.twoD:
                        s_prof   = np.copy(integrator.fd2['g'][:,0])
                        integrator.fd2.set_scales(int(np.ceil(4096/len(s_prof))), keep_data=True)
                        s_prof_big = np.copy(integrator.fd2['g'][:,0])
                        z_big      = integrator.domain.grid(0, scales=int(np.ceil(4096/len(s_prof)))).flatten()
                    else:
                        s_prof   = np.copy(integrator.fd2['g'][0,:])
                        integrator.fd2.set_scales(int(np.ceil(4096/len(s_prof))), keep_data=True)
                        s_prof_big = np.copy(integrator.fd2['g'][0,:])
                        z_big      = integrator.domain.grid(1, scales=int(np.ceil(4096/len(s_prof)))).flatten()

                    therm_z = integrator.z.flatten()[s_prof   < 0.2*s_prof.min()]

                    profile_z = s_prof - 0.2*s_prof.min()
                    z_min = z_big[s_prof_big.argmin()]
                    vortex_height[count - 1] = z_min

                    #find radius at that height
                    integrator.fd1['g'] = integrator.rho0['g']*s1[i,:,:]
                    integrator.fd1.integrate('z', out=integrator.fd3)
#                    integrator.fd1.interpolate(z=vortex_height[count-1], out=integrator.fd3)
                    integrator.fd3.set_scales(1, keep_data=True)
                    if self.twoD:
                        s_prof_r = np.copy(integrator.fd3['g'][0,:])
                        integrator.fd3.set_scales(int(np.ceil(2048/len(s_prof_r))), keep_data=True)
                        s_prof_r_big = np.copy(integrator.fd3['g'][0,:])
                        r_big      = integrator.domain.grid(1, scales=int(np.ceil(2048/len(s_prof_r)))).flatten()
                    else:
                        s_prof_r = np.copy(integrator.fd3['g'][:,0])
                        integrator.fd3.set_scales(int(np.ceil(2048/len(s_prof_r))), keep_data=True)
                        s_prof_r_big = np.copy(integrator.fd3['g'][:,0])
                        r_big      = integrator.domain.grid(0, scales=int(np.ceil(2048/len(s_prof_r)))).flatten()

                    profile_r = s_prof_r - 0.3*s_prof_r.min()
                    r_min     = r_big[s_prof_r_big.argmin()]
                    vortex_radius[count - 1] = r_min

                    profile_z = s_prof - 0.1*s_prof.min()
                    z_f2 = interp1d(integrator.z.flatten(), profile_z, fill_value='extrapolate')
                    try:
                        this_z_cb = brentq(z_f2, 0, vortex_height[count-1])
                    except:
                        this_z_cb = integrator.z.flatten()[profile_z < profile_z.min()*0.1][0]

                    therm_bot, therm_top, this_Rz = therm_z[0], therm_z[-1], (therm_z[-1]-therm_z[0])/2

                    vortex_rho[count - 1]    = (1 + self.grad_T_ad*(vortex_height[count - 1] - self.Lz))**self.m_ad
                    vortex_T[count - 1]      = (1 + self.grad_T_ad*(vortex_height[count - 1] - self.Lz))

                    z_cb[count - 1] = this_z_cb
                    logger.info('cb calc count {}, t = {:.2e} / z_cb {:.2e} / (r,z) = ({:.2e}, {:.2e})'.format(count, self.times[count-1], this_z_cb, vortex_radius[count-1], vortex_height[count-1]))

                    mask = (np.sqrt(rr**2 + (zz-vortex_height[count - 1])**2) > 2*vortex_radius[count - 1])
                    #Circulation
                    Vort = V[i,:]
                    Vort[mask] *= 0
                    integrator.fd1['g'] = Vort 
                    circ[count-1] = Gamma = integrator.full_integrate()
                    integrator.field_clean()
                    #Buoyancy
                    Buoy = s1[i,:]
                    Buoy[mask] *= 0
                    integrator.fd1['g'] = (2*np.pi*integrator.rho0['g']*Buoy*integrator.r)
                    B[count-1] = this_B = integrator.full_integrate()
                    integrator.field_clean()
                    #momentum
                    Vel = w[i,:]
                    Vel[mask] *= 0
                    integrator.fd1['g'] = (2*np.pi*integrator.rho0['g']*Vel*integrator.r)
                    momentum[count-1] = P = integrator.full_integrate()
                    integrator.field_clean()

                    logger.info('B: {:.2e}, Gamma: {:.2e}, P: {:.2e}\n\n'.format(this_B, Gamma, P))

            vortex_dT_dt = differentiate(self.times, vortex_T)#np.diff(vortex_T)/np.diff(times)
            vortex_w = vortex_dT_dt/self.grad_T_ad


            output_file = h5py.File('{:s}/z_cb_file.h5'.format(self.full_out_dir), 'w')
            output_file['times']            = self.times
            output_file['z']                = z
            output_file['z_cb']             = z_cb
            output_file['vortex_height']    = vortex_height
            output_file['vortex_w']         = vortex_w
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


    def measure_values_in_thermal(self):
        """
        After the thermal contour is found, this analyzes the properties of the thermal in depth
        by taking integrals over the dedalus domain. Integrals are output into
        root_dir/out_dir/post_analysis.h5
        """
        f = h5py.File('{:s}/z_cb_file.h5'.format(self.full_out_dir), 'r')
        w_cb = np.zeros_like(self.times)
        vortex_w = f['vortex_w'].value
        w_cb[2:-2] = vortex_w
        f.close()

        c_f = h5py.File('{:s}/contour_file.h5'.format(self.full_out_dir), 'r')
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
                            'full_impulse',
                            'full_momentum',
                            'full_circ',
                            'full_rho_s1',
                            'full_ke',
                            'full_pe']
        integs = dict()
        for f in integral_fields:
            integs[f] = np.zeros(len(self.times))
        radius = np.zeros_like(self.times)
        height = np.zeros_like(self.times)
        c_f = h5py.File('{:s}/contour_file.h5'.format(self.full_out_dir), 'r')
        contours = c_f['contours'].value
        c_f.close()
        count = 0
        for filenum, fn in enumerate(self.files):
            #Read entropy, density, velocity, vorticity
            r, z, rr, zz, s1, w, u, V = self.read_file(fn)
            f    = h5py.File(self.prof_files[filenum], 'r')
            time = f['scales']['sim_time'].value
            f.close()

            if filenum == 0:
                integrator = DedalusIntegrator(len(r), len(z), self.Lr, self.Lz, self.grad_T_ad, self.m_ad, r_cheby=self.twoD, twoD=self.twoD)

            for i in range(s1.shape[0]):
                count += 1
                logger.info('analyze count {}'.format(count))

                contour = contours[count-1,:]
                if contour.max() > 0:
                    radius[count-1] = contour.max()
                    height[count-1] = z[contour.argmax()]
                therm_mask = np.zeros_like(rr, dtype=bool)
                for k in range(len(z)):
                    if self.twoD:
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
                            'int_pe' : 2*np.pi*s1[i,:]*integrator.rho0['g']*self.g*-1*integrator.z*integrator.r,
                            'int_impulse' : np.pi*integrator.rho0['g']*V[i,:]*integrator.r**2,
                            'int_rho_s1' : 2*np.pi*integrator.rho0['g']*s1[i,:]*integrator.r, 
                            'int_rho_s1_above' : 2*np.pi*integrator.rho0['g']*s1[i,:]*integrator.r, 
                            'int_w' : 2*np.pi*w[i,:]*integrator.r, 
                            }

                integrands['full_impulse'] = np.copy(integrands['int_impulse'])
                integrands['full_momentum'] = np.copy(integrands['int_mom'])
                integrands['full_circ'] = np.copy(integrands['int_circ'])
                integrands['full_rho_s1'] = np.copy(integrands['int_rho_s1'])
                integrands['full_ke'] = np.copy(2*np.pi*integrator.rho0['g']*(u[i,:]**2+w[i,:]**2)/2*integrator.r)
                integrands['full_pe'] = np.copy(integrands['int_pe'])


                for name, values in integrands.items():
                    if 'above' in name: 
                        integrator.fd1['g'][zz > contour_top] = 1.*values[zz > contour_top]
                    elif 'full_' in name:
                        integrator.fd1['g'] = values
                    else:
                        integrator.fd1['g'][therm_mask] = 1.*values[therm_mask]
                    integs[name][count-1] = integrator.full_integrate()
                    integrator.field_clean()

                if np.sum(therm_mask) > 0:
                    integs['max_s1_therm'][count-1]  = np.max(np.abs(s1[i,:][therm_mask]))
                    s1_in_therm = s1[i,:][therm_mask]
                    mask = therm_mask*(s1[i,:] < 0.1*s1_in_therm.min())
                    integrator.fd1['g'][mask] = 1.*rr[mask]
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


        output_file = h5py.File('{:s}/post_analysis.h5'.format(self.full_out_dir), 'w')
        output_file['times']       = self.times

        for f, fd in integs.items():
            output_file[f]    = fd
        output_file['height']      = height
        output_file['radius']      = radius
        output_file.close()

    def plot_colormeshes(self):
        """
        Plot colormeshes of the thermal evolution.  The azimuthally averaged values of
        entropy, velocity (radial and vertical), and vorticity are plotted. The outline
        of the thermal, as found by the thermal tracker, is overplotted if available.
        """
        if os.path.exists('{:s}/fit_file.h5'.format(self.full_out_dir)):
            c_f = h5py.File('{:s}/fit_file.h5'.format(self.full_out_dir), 'r')
            z_cb = c_f['z_cb'].value
            radius = c_f['vortex_radius'].value
            height = c_f['vortex_height'].value
            c_f.close()
        else:
            radius, height, z_cb = np.zeros_like(self.times), np.zeros_like(self.times), np.zeros_like(self.times)
        if os.path.exists('{:s}/contour_file.h5'.format(self.full_out_dir)):
            c_f = h5py.File('{:s}/contour_file.h5'.format(self.full_out_dir), 'r')
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

        for fn, fp in zip(self.files, self.prof_files):
            #Read entropy, density, velocity, vorticity
            r, z, rr, zz, s1, w, u, V = self.read_file(fn)
            f    = h5py.File(fp, 'r')
            time = f['scales']['sim_time'].value
            f.close()
            
            if contours is None:
                contours = np.zeros((len(self.times),len(z)))


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
                    maxv = np.sort(np.abs(fd).flatten())[int(0.998*len(fd.flatten()))]
                    cmesh = ax.pcolormesh(rr, zz, fd, cmap='RdBu_r', vmin=-maxv, vmax=maxv)
                    cbar = fig.colorbar(cmesh, cax=cax, orientation="horizontal", ticks=[-maxv*.75,0,maxv*.75])
                    cax.set_xticklabels(['', r'$\pm{:.2e}$'.format(maxv), ''])
                    cax.xaxis.set_ticks_position('none')
                    cax.set_title(title)
                    ax.set_xlim(0, 0.25*self.Lz)
                    ax.set_ylim(0, self.Lz)
                    ax.axhline(z_cb[count-1], lw=0.5, c='k', ls='--')
                    ax.scatter(radius[count-1], height[count-1], color='black', s=2)
                    ax.plot(contour[good_contour], z[good_contour], c='k', lw=0.5)
                    ax.axhline(contour_top, c='grey', lw=0.5)
                    if j > 0: ax.set_yticks([])

                plt.suptitle('sim_time = {:.4e}'.format(self.times[count-1]))
                fig.savefig('{:s}/radial_fields_{:04d}.png'.format(self.full_out_dir,count), dpi=150, figsize=(10,6))
                [ax.cla() for ax in axs]
                [cax.cla() for cax in caxs]

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
    plt.xlabel(labels[0])

    return fig, (ax1, ax2), best_fit, fits




