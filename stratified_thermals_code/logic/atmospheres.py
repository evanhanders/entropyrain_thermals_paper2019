import numpy as np
from collections import OrderedDict

import logging
logger = logging.getLogger(__name__)

from dedalus import public as de

def set_field(field, in_arr, scales=1):
    """
    Sets the values in a dedalus field to match those in an input NumPy array
    at the proper scale. Returns the field at that scale.

    Parameters
    ----------
    field   : A Dedalus Field object
        The field to set the values of
    in_arr  : A NumPy array
        The NumPy array to set the field values to
    scales  : float, optional
        The size scale at which to set the field.
    """
    field.set_scales(scales, keep_data=True)
    field['g'] = in_arr
    field.set_scales(scales, keep_data=True)
    return field



class IdealGasAtmosphere:
    """
    An abstract class which contains many of the attributes of an
    ideal gas atmosphere, to be extended to more specific cases.

    Attributes:
    -----------
        params      : OrderedDict 
            Scalar atmospheric parameters (floats, ints)
        gamma       : Float
            The adiabatic index of the atmosphere
        Cp, Cv      : Floats
            Specific heat at constant pressure, volume
        g           : Float
            Gravity, constant. (default value sets T_ad_z = -1)
        R           : Float
            The ideal gas constant
        T_ad_z      : Float
            The adiabatic temperature gradient, as defined in Anders&Brown2017 & elsewhere
        de_domain   : A DedalusDomain object 
            The domain on which the atmosphere will be built
        atmo_fields : OrderedDict 
            contains fields for the variables: T0, T0_z, T0_zz, rho0, ln_rho0, ln_rho0_z, phi
    """

    def __init__(self, de_domain, gamma=5./3, R=1, g=None):
        """
        Initialize the atmosphere. Inputs that match attributes in the class docstring are
        defined there.
        """
        self.params = OrderedDict()
        self.rho_fit     = None
        self.params['R'] = R
        self.params['gamma'] = gamma
        self.params['Cp'] = gamma*R/(gamma-1.)
        self.params['Cv'] = self.params['Cp'] - R
        if g is None:
            self.params['g'] = self.params['Cp']
        else:
            self.params['g'] = g
        self.params['T_ad_z'] = - self.params['g'] / self.params['Cp']

        self.de_domain = de_domain
        self.atmo_fields = OrderedDict()
        fds = ['T0', 'T0_z', 'rho0', 'ln_rho0', 'ln_rho0_z', 'phi', 'xi', 'xi_z', 'xi_L', 'xi_R']
        for f in fds:
            self.atmo_fields[f] = self.de_domain.new_ncc()

    def _setup_atmosphere(self, *args, **kwargs):
        pass

class Polytrope(IdealGasAtmosphere):
    """
    An extension of an IdealGasAtmosphere for a polytropic stratification
    """
    def __init__(self, de_domain, epsilon=0, n_rho=3, aspect_ratio=4, **kwargs):
        """
        Initialize the polytrope. Additional parameters

        Parameters:
        -----------
        de_domain    : A DedalusDomain object
            The domain on which the atmosphere is built
        epsilon      : Float
            The superadiabatic excess of the atmosphere (see Anders&Brown2017)
        n_rho        : Float
            The number of density scale heights of the atmosphere
        aspect_ratio : Float
            The aspect ratio of the atmosphere (Lx/Lz or Ly/Lz)
        """
        super(Polytrope, self).__init__(de_domain, **kwargs)
        self.params['m_ad']     = 1/(self.params['gamma'] - 1)
        self.params['m']        = self.params['m_ad'] - epsilon
        self.params['n_rho']    = n_rho
        self.params['Lz']       = np.exp(self.params['n_rho']/self.params['m']) - 1
        self.params['Lx']       = aspect_ratio*self.params['Lz']
        self.params['Ly']       = aspect_ratio*self.params['Lz']
        self._setup_atmosphere()

    def _setup_atmosphere(self):
        """
        Sets up all atmospheric fields (T0, T0_z, T0_zz, etc.) according to a 
        polytropic stratification of the form:

            T0 = (Lz + 1 - z)
            rho0 = T0**m 
        """
        self.atmo_fields['T0_zz'] = self.de_domain.new_ncc()
        T0 = (self.params['Lz'] + 1 - self.de_domain.z)
        self.atmo_fields['T0'] = set_field(self.atmo_fields['T0'], T0, scales=1)
        self.atmo_fields['T0_z'] = set_field(self.atmo_fields['T0_z'], -1, scales=1)
        self.atmo_fields['T0_zz'] = set_field(self.atmo_fields['T0_zz'], 0, scales=1)
        
        rho0 = T0**self.params['m']
        self.atmo_fields['rho0'] = set_field(self.atmo_fields['rho0'], rho0, scales=1)

        ln_rho0 = np.log(rho0)
        self.atmo_fields['ln_rho0'] = set_field(self.atmo_fields['ln_rho0'], ln_rho0, scales=1)
        self.atmo_fields['ln_rho0'].differentiate('z', out=self.atmo_fields['ln_rho0_z'])

        phi = -self.params['g']*(1 + self.params['Lz'] - self.de_domain.z)
        self.atmo_fields['phi'] = set_field(self.atmo_fields['phi'], phi, scales=1)
