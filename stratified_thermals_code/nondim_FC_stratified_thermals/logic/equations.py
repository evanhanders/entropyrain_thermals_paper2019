import numpy as np
from collections import OrderedDict

import logging
logger = logging.getLogger(__name__)

from dedalus import public as de
from mpi4py import MPI


class FullyCompressibleEquations:
    """
    An abstract class containing the fully compressible equations which must
    be extended to specify the type of diffusivities

    Attributes
    ----------
    de_domain   : A DedalusDomain object
        The domain on which the equations are solved
    atmosphere  : An IdealGasAtmosphere object
        The atmosphere in which the equations are solved
    variables   : list
        A list of strings of problem variables.
    problem     : A Dedalus Problem object
        The problem in which the equations are solved.
    """
    def __init__(self, de_domain, atmosphere):
        """
        Construct the equation class. Inputs are described in the class docstring.
        """
        self.de_domain = de_domain
        self.atmosphere = atmosphere
        self.variables = ['u','u_z','v', 'v_z', 'w','w_z','T1', 'T1_z', 'ln_rho1']
        if not self.de_domain.threeD:
            self.variables.remove('v')
            self.variables.remove('v_z')

        self.problem = de.IVP(self.de_domain.domain, variables=self.variables, ncc_cutoff=1e-6)
        if not self.de_domain.threeD:
            self.problem.substitutions['v'] = '0'
            self.problem.substitutions['v_z'] = '0'
            self.problem.substitutions['dy(A)'] = '0*A'

    def set_equations(self, *args, **kwargs):
        """ 
        Sets the fully compressible equations of in a ln_rho / T formulation. These
        equations take the form:
        
        D ln ρ + ∇ · u = 0
        D u = - ∇ T - T∇ ln ρ - gẑ + (1/ρ) * ∇ · Π
        D T + (γ - 1)T∇ · u - (1/[ρ Cv]) ∇ · (- Kap∇ T) = (1/[ρ Cv])(Π ·∇ )·u 
        
        Where

        D = ∂/∂t + (u · ∇ ) 

        and

        Π = - Mu (∂u_i/∂x_j + ∂u_j/∂x_i - (2/3)D_{ij}∇ · u)

        is the viscous stress tensor. The variables are u (vector velocity), T (temp) and 
        ρ (density). Temperature, density, and pressure are related through an ideal gas
        equation of state,

        P = ρT

        Which has already been assumed in the formulation of these equations.
        """

        self.set_params_and_subs(*args, **kwargs)

        self.problem.add_equation(    "dz(u) - u_z = 0")
        if self.de_domain.threeD:
            self.problem.add_equation("dz(v) - v_z = 0")
        self.problem.add_equation(    "dz(w) - w_z = 0")
        self.problem.add_equation(    "dz(T1) - T1_z = 0")
        self.problem.add_equation((    "(scale_c)*( dt(ln_rho1)   + epsilon**(-1)*(w*ln_rho0_z + Div_u) ) = (scale_c)*(-UdotGrad(ln_rho1, dz(ln_rho1)))"))
        self.problem.add_equation(    ("(scale_m_z)*( dt(w) + (T1_z     + T0*dz(ln_rho1) + T1*ln_rho0_z)/(-T_ad_z) - L_visc_w) = "
                                       "(scale_m_z)*(- UdotGrad(w, w_z) - epsilon*T1*dz(ln_rho1)/(-T_ad_z) + R_visc_w)"))
        self.problem.add_equation(    ("(scale_m)*( dt(u) + (dx(T1)   + T0*dx(ln_rho1))/(-T_ad_z)                  - L_visc_u) = "
                                       "(scale_m)*(-UdotGrad(u, u_z) - epsilon*T1*dx(ln_rho1)/(-T_ad_z) + R_visc_u)"))
        if self.de_domain.threeD:
            self.problem.add_equation(("(scale_m)*( dt(v) + (dy(T1)   + T0*dy(ln_rho1))/(-T_ad_z)                  - L_visc_v) = "
                                       "(scale_m)*(-UdotGrad(v, v_z) - epsilon*T1*dy(ln_rho1)/(-T_ad_z) + R_visc_v)"))
        self.problem.add_equation((    "(scale_e)*( dt(T1)   + epsilon**(-1)*(w*T0_z  + (gamma-1)*T0*Div_u) -  L_thermal) = "
                                       "(scale_e)*(-UdotGrad(T1, T1_z) - (gamma-1)*T1*Div_u + R_thermal + R_visc_heat + source_terms)"))

    def set_BC(self):
        """ 
        Fixed Temperature, Impenetrable, stress-free boundaries are set at 
        the top and bottom of the domain.
        """
        self.problem.add_bc('left(T1) = 0')
        self.problem.add_bc('right(T1) = 0')
        self.problem.add_bc('left(u_z) = 0')
        self.problem.add_bc('right(u_z) = 0')
        if self.de_domain.threeD:
            self.problem.add_bc('left(v_z) = 0')
            self.problem.add_bc('right(v_z) = 0')
            self.problem.meta['v_z']['z']['dirichlet'] = True
        self.problem.add_bc('left(w) = 0')
        self.problem.add_bc('right(w) = 0')

        self.problem.meta['T1','u_z','w']['z']['dirichlet'] = True

    def set_params_and_subs(self, *args, **kwargs):
        """ 
        Set up all important thermodynamic and atmospheric quantities, as well as
        substitutions related to these quantities.
        """
        for k, item in self.atmosphere.params.items():
            self.problem.parameters[k] = item
        for k, item in self.atmosphere.atmo_fields.items():
            self.problem.parameters[k] = item

        self._setup_diffusivities(*args, **kwargs)
        self.problem.substitutions['Cv_inv']   = '(1/Cv)'
        self.problem.substitutions['rho_full'] = '(rho0*exp(epsilon*ln_rho1))'
        self.problem.substitutions['rho_fluc'] = '(rho0*(exp(epsilon*ln_rho1)-1))'
        self.problem.substitutions['T_full']   = '(T0 + epsilon*T1)'
        self.problem.substitutions['p']        = '(R*rho_full*T_full)'
        self.problem.substitutions['p0']       = '(R*rho0*T0)'
        self.problem.substitutions['p1']       = '(p - p0)'
        self.problem.substitutions['vel_rms']  = 'sqrt(u**2 + v**2 + w**2)'
        self.problem.substitutions['Ma_rms']   = '(sqrt(epsilon*(-T_ad_z))*vel_rms/sqrt(T_full))'
        self.problem.substitutions['s1']       = 'Cp*(1/gamma*log(1+epsilon*T1/T0) - (gamma-1)/gamma*epsilon*ln_rho1)/epsilon'

        self._set_diffusion_subs(**kwargs)
        self.problem.substitutions['Re_rms']   = '(full_Re*vel_rms)'
        self.problem.substitutions['Pe_rms']   = '(full_Pe*vel_rms)'

    def _set_diffusion_subs(self):
        pass

    def _setup_diffusivities(self):
        pass

    def set_operators(self):
        """
        Set operator substitutions in the problem
        """
        self.problem.substitutions['Lap(f, f_z)'] = "(dx(dx(f)) + dy(dy(f)) + dz(f_z))"
        self.problem.substitutions['Div(fx, fy, fz_z)'] = "(dx(fx) + dy(fy) + fz_z)"
        self.problem.substitutions['Div_u'] = "Div(u, v, w_z)"
        self.problem.substitutions['UdotGrad(f, f_z)'] = "(u*dx(f) + v*dy(f) + w*(f_z))"
        if self.de_domain.threeD:
            self.problem.substitutions['mid_interp(A)'] = 'interp(interp(A, x=0), y=0)'
            self.problem.substitutions['plane_avg(A)']  = 'integ(A, "x", "y")/Lx/Ly'
            self.problem.substitutions['vol_avg(A)']    = 'integ(A)/Lx/Ly/Lz'
        else:
            self.problem.substitutions['mid_interp(A)'] = 'interp(A, x=0)'
            self.problem.substitutions['plane_avg(A)']  = 'integ(A, "x")/Lx'
            self.problem.substitutions['vol_avg(A)']    = 'integ(A)/Lx/Lz'

        self.problem.substitutions["Sxx"] = "(2*dx(u) - 2/3*Div_u)"
        self.problem.substitutions["Syy"] = "(2*dy(v) - 2/3*Div_u)"
        self.problem.substitutions["Szz"] = "(2*w_z   - 2/3*Div_u)"
        self.problem.substitutions["Sxy"] = "(dx(v) + dy(u))"
        self.problem.substitutions["Sxz"] = "(dx(w) +  u_z )"
        self.problem.substitutions["Syz"] = "(dy(w) +  v_z )"

        self.problem.substitutions['Vort_x'] = '(dy(w) - v_z)'
        self.problem.substitutions['Vort_y'] = '( u_z  - dx(w))'
        self.problem.substitutions['Vort_z'] = '(dx(v) - dy(u))'
        self.problem.substitutions['enstrophy']   = '(Vort_x**2 + Vort_y**2 + Vort_z**2)'


class KappaMuFCE(FullyCompressibleEquations):
    """
    An extension of the fully compressible equations where the diffusivities are
    set based on kappa and mu, not chi and nu.
    """

    def __init__(self, de_domain, atmosphere, *args, **kwargs):
        """
        Construct the atmosphere.  See parent class init for some input defns.
        """
        super(KappaMuFCE, self).__init__(de_domain, atmosphere, *args, **kwargs)

    def _set_diffusion_subs(self, viscous_heating=True):
        """
        Setup substitutions in the momentum and energy equations.
       
        Parameters
        ----------
        viscous_heating : bool
            If True, include viscous heating. If False, zero out viscous heating term.
        """
        self.problem.substitutions['full_Re']    = '(Re*rho_full)'
        self.problem.substitutions['full_Pe']    = '(Pe*rho_full)'
        self.problem.substitutions['scale_m_z']  = '(T0)'
        self.problem.substitutions['scale_m']    = '(T0)'
        self.problem.substitutions['scale_e']    = '(T0)'
        self.problem.substitutions['scale_c']    = '(T0)'
        self.problem.substitutions['rho_inv_approx'] = '(1/T0)'

        #Viscous subs -- momentum equation     
        self.problem.substitutions['visc_u']   = "( (Re**(-1))*(Lap(u, u_z) + 1/3*Div(dx(u), dx(v), dx(w_z))))"
        self.problem.substitutions['visc_v']   = "( (Re**(-1))*(Lap(v, v_z) + 1/3*Div(dy(u), dy(v), dy(w_z))))"
        self.problem.substitutions['visc_w']   = "( (Re**(-1))*(Lap(w, w_z) + 1/3*Div(  u_z, dz(v), dz(w_z))))"                
        self.problem.substitutions['L_visc_u'] = "(rho_inv_approx*visc_u)"
        self.problem.substitutions['L_visc_v'] = "(rho_inv_approx*visc_v)"
        self.problem.substitutions['L_visc_w'] = "(rho_inv_approx*visc_w)"                
        self.problem.substitutions['R_visc_u'] = "(visc_u/rho_full - L_visc_u)"
        self.problem.substitutions['R_visc_v'] = "(visc_v/rho_full - L_visc_v)"
        self.problem.substitutions['R_visc_w'] = "(visc_w/rho_full - L_visc_w)"

        self.problem.substitutions['thermal'] = ('( (Cv_inv/Pe)*(Lap(T1, T1_z)) )')
        self.problem.substitutions['L_thermal'] = ('(rho_inv_approx*thermal)')
        self.problem.substitutions['R_thermal'] = ('( thermal/rho_full - (L_thermal) )' )
        self.problem.parameters['source_terms'] = 0

        #Viscous heating
        if viscous_heating:
            self.problem.substitutions['R_visc_heat'] = " ((-T_ad_z)*Re**(-1)/rho_full*Cv_inv)*(dx(u)*Sxx + dy(v)*Syy + w_z*Szz + Sxy**2 + Sxz**2 + Syz**2)"
        else:
            self.problem.parameters['R_visc_heat'] = 0


    def _setup_diffusivities(self, Reynolds, Prandtl, epsilon, radius, **kwargs):
        """
        Setup the diffusivities based on a few input values. Kappa and Mu are
        constants.

        Parameters
        ----------
        Reynolds    : float
            The estimated reynolds number used in setting the diffusivities.
        Prandtl     : float
            The prandtl number of the simulations (viscous / thermal diffusivity)
        epsilon     : float
            The estimated magnitude of the entropy signature that will drive flows.
        radius      : float
            Half the length scale to use in definining viscosity from the Reynolds number.
        """
        self.problem.parameters['Re'] = Reynolds
        self.problem.parameters['Pe'] = Reynolds*Prandtl
