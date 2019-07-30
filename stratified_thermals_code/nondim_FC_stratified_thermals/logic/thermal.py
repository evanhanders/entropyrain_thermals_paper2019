import numpy as np
from collections import OrderedDict

import logging
logger = logging.getLogger(__name__)
from scipy.special import erf

from mpi4py import MPI

class Thermal:
    """
    A class that creates a thermal in a Fully Compressible simulation

    Attributes
    ----------
    de_domain   : DedalusDomain object
        The domain in which the thermal experiment is being done
    atmosphere  : An IdealGasAtmosphere object
        The atmosphere in which the thermal is falling.
    falling     : bool
        If True, do a cold, falling thermal. Else, do a hot, rising thermal.
    radius      : float
        The size of the thermal's radius, in simulation units.
    r_width     : float
        The sharpness of the edges of the thermal in the initial conditions.
    A0          : float
        The magnitude of the initial thermal's entropy perturbation.
    z_pert      : float
        simulation z-location of perturbation
    """

    def __init__(self, de_domain, atmosphere, falling=False, radius=1, r_width=None, 
                 A0=1e-4, z_pert = None):
        """
        Initialize the thermal class. Inputs are described in the class attributes in the
        class docstring.
        """
        self.de_domain     = de_domain
        self.atmosphere = atmosphere
        self.radius     = radius
        if r_width is None:
            self.r_width = radius/5
        else:
            self.r_width = r_width
        self.falling    = falling
        self.A0         = A0

        if self.falling:
            self.sign   = -1
        else:
            self.sign   = 1
       
        if z_pert is None:
            if self.falling:
                self.z_pert = self.de_domain.Lz - 3*self.radius
            else:
                self.z_pert = 3*self.radius
        else:
            self.z_pert = z_pert


    def set_thermal(self, T1, T1_z, ln_rho1):
        """
        Put a thermal into the intial conditions of a simulation. The temperature,
        temperature derivative, and log density fields from the problem solver must be
        provided.
        
        For pressure equilibrium, rho_fluc * T_fluc = 1, so P0 = rho0*exp(ln_rho1)*(T0+T1), so
         (1 + T1/T0) = exp(-ln_rho1). Plugging this back into the entropy equation, we find that
         to get a specific entropy perturbation of magnitude epsilon, the ln_rho1 perturbation
         should be -epsilon/Cp.

        Parameters:
        -----------
        T1, T1_z, ln_rho1   : Dedalus Field objects
            The fluctuations of temperature, temperature derivative, and log density in the solver.
        """
        T0_pert = np.mean(self.atmosphere.atmo_fields['T0'].interpolate(z=self.z_pert)['g'])

        x, y, z = self.de_domain.x, self.de_domain.y, self.de_domain.z
        if y is None: y = 0
        r = np.sqrt(x**2+y**2+(z-self.z_pert)**2)   
        initial_perturbation = (1-erf( (r - self.radius)/self.r_width))/2 # positively signed bump function

        #Set perturbation
        S1 = self.sign*initial_perturbation
        ln_rho1['g'] = -1*S1/self.atmosphere.params['Cp']
        T0 = self.atmosphere.atmo_fields['T0']
        T0.set_scales(1, keep_data=True)
        ln_rho1.set_scales(1, keep_data=True)
        T1['g'] = T0['g']*(np.exp(-self.A0*ln_rho1['g']) - 1)/self.A0


