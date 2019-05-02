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

        self.problem = de.IVP(self.de_domain.domain, variables=self.variables)
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
        self.problem.add_equation((    "(scale_c)*( dt(ln_rho1)   + w*ln_rho0_z + Div_u ) = (scale_c)*(-UdotGrad(ln_rho1, dz(ln_rho1)))"))
        self.problem.add_equation(    ("(scale_m_z)*( dt(w) + T1_z     + T0*dz(ln_rho1) + T1*ln_rho0_z - L_visc_w) = "
                                       "(scale_m_z)*(- UdotGrad(w, w_z) - T1*dz(ln_rho1) + R_visc_w)"))
        self.problem.add_equation(    ("(scale_m)*( dt(u) + dx(T1)   + T0*dx(ln_rho1)                  - L_visc_u) = "
                                       "(scale_m)*(-UdotGrad(u, u_z) - T1*dx(ln_rho1) + R_visc_u)"))
        if self.de_domain.threeD:
            self.problem.add_equation(("(scale_m)*( dt(v) + dy(T1)   + T0*dy(ln_rho1)                  - L_visc_v) = "
                                       "(scale_m)*(-UdotGrad(v, v_z) - T1*dy(ln_rho1) + R_visc_v)"))
        self.problem.add_equation((    "(scale_e)*( dt(T1)   + w*T0_z  + (gamma-1)*T0*Div_u -  L_thermal) = "
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
        self.problem.substitutions['rho_full'] = '(rho0*exp(ln_rho1))'
        self.problem.substitutions['rho_fluc'] = '(rho0*(exp(ln_rho1)-1))'
        self.problem.substitutions['T_full']   = '(T0 + T1)'
        self.problem.substitutions['p']        = '(R*rho_full*T_full)'
        self.problem.substitutions['p0']       = '(R*rho0*T0)'
        self.problem.substitutions['p1']       = '(p - p0)'
        self.problem.substitutions['vel_rms']  = 'sqrt(u**2 + v**2 + w**2)'
        self.problem.substitutions['Ma_rms']   = '(vel_rms/sqrt(T_full))'
        self.problem.substitutions['s1']       = '(1/gamma*log(1+T1/T0) - (gamma-1)/gamma*ln_rho1)' #technically s1/cp

        self._set_diffusion_subs(**kwargs)
        self.problem.substitutions['Re_rms']   = '(vel_rms*L/nu)'
        self.problem.substitutions['Pe_rms']   = '(vel_rms*L/chi)'

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

    def __init__(self, de_domain, atmosphere, A=0.5, B=0.5, *args, **kwargs):
        """
        Construct the atmosphere.  See parent class init for some input defns.

        Parameters
        ----------
        A, B   : Floats
            Used in approximating rho^{-1} = (A/T0 + B/T0**2) to reduce bandwidth.
        """
        super(KappaMuFCE, self).__init__(de_domain, atmosphere, *args, **kwargs)
        self.problem.parameters['A'] = A
        self.problem.parameters['B'] = B

    def _set_diffusion_subs(self, viscous_heating=True):
        """
        Setup substitutions in the momentum and energy equations.
       
        Parameters
        ----------
        viscous_heating : bool
            If True, include viscous heating. If False, zero out viscous heating term.
        """
        self.problem.substitutions['nu']   =     '(Mu/rho_full)'
        self.problem.substitutions['chi']  =     '(Kap/rho_full)'
        self.problem.substitutions['scale_m_z']  = '(T0)'
        self.problem.substitutions['scale_m']    = '(T0)'
        self.problem.substitutions['scale_e']    = '(T0)'
        self.problem.substitutions['scale_c']    = '(T0)'
        self.problem.substitutions['rho_inv_approx'] = '(1/T0)'

        #Viscous subs -- momentum equation     
        self.problem.substitutions['visc_u']   = "( (Mu)*(Lap(u, u_z) + 1/3*Div(dx(u), dx(v), dx(w_z))) + (Mu_z)*(Sxz))"
        self.problem.substitutions['visc_v']   = "( (Mu)*(Lap(v, v_z) + 1/3*Div(dy(u), dy(v), dy(w_z))) + (Mu_z)*(Syz))"
        self.problem.substitutions['visc_w']   = "( (Mu)*(Lap(w, w_z) + 1/3*Div(  u_z, dz(v), dz(w_z))) + (Mu_z)*(Szz))"                
        self.problem.substitutions['L_visc_u'] = "(rho_inv_approx*visc_u)"
        self.problem.substitutions['L_visc_v'] = "(rho_inv_approx*visc_v)"
        self.problem.substitutions['L_visc_w'] = "(rho_inv_approx*visc_w)"                
        self.problem.substitutions['R_visc_u'] = "(visc_u/rho_full - L_visc_u)"
        self.problem.substitutions['R_visc_v'] = "(visc_v/rho_full - L_visc_v)"
        self.problem.substitutions['R_visc_w'] = "(visc_w/rho_full - L_visc_w)"

        self.problem.substitutions['thermal'] = ('( (Cv_inv)*(Kap*Lap(T1, T1_z) + Kap_z*T1_z) )')
        self.problem.substitutions['L_thermal'] = ('(rho_inv_approx*thermal)')
        self.problem.substitutions['R_thermal'] = ('( thermal/rho_full - (L_thermal) + (Cv_inv/(rho_full))*(Kap*T0_zz + Kap_z*T0_z) )' )
        self.problem.parameters['source_terms'] = 0

        #Viscous heating
        if viscous_heating:
            self.problem.substitutions['R_visc_heat'] = " (Mu/rho_full*Cv_inv)*(dx(u)*Sxx + dy(v)*Syy + w_z*Szz + Sxy**2 + Sxz**2 + Syz**2)"
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
        L   = 2*radius
        self.problem.parameters['L'] = L
        mu_top = nu_top = np.sqrt(self.atmosphere.params['g']*L**3*epsilon/self.atmosphere.params['Cp'])/Reynolds
        kappa_top = mu_top/Prandtl
        
        self.kappa       = self.de_domain.new_ncc()
        self.mu          = self.de_domain.new_ncc()
        self.kappa['g']  = kappa_top
        self.mu['g']     = mu_top

        self.problem.parameters['Kap'] = self.kappa
        self.problem.parameters['Mu']  = self.mu
        self.problem.substitutions['Kap_z']  = '0'
        self.problem.substitutions['Mu_z']  = '0'


class ChiNuFCE(FullyCompressibleEquations):
    """
    An extension of the fully compressible equations where the diffusivities are
    set based on the diffusivities, nu and chi.
    """

    def __init__(self, domain, atmosphere, *args, **kwargs):
        """
        Construct the atmosphere.  See parent class init for some input defns.
        """
        super(ChiNuFCE, self).__init__(domain, atmosphere, *args, **kwargs)

    def _set_diffusion_subs(self, viscous_heating=True):
        """
        Setup substitutions in the momentum and energy equations. Assumes that the
        background source terms in the energy equations do not adjust the background
        atmosphere (diffusivity acting on the background grad T0 * grad ln rho_0 term
        is zero).
       
        Parameters
        ----------
        viscous_heating : bool
            If True, include viscous heating. If False, zero out viscous heating term.
        """
        self.problem.substitutions['scale_m_z']  = '(T0)'
        self.problem.substitutions['scale_m']    = '(T0)'
        self.problem.substitutions['scale_e']    = '(T0)'
        self.problem.substitutions['scale_c']    = '(T0)'


        self.viscous_term_u = " (nu*(Lap(u, u_z) + 1/3*Div(dx(u), dx(v), dx(w_z))) + (nu*ln_rho0_z + nu_z) * Sxz) "
        self.viscous_term_v = " (nu*(Lap(v, v_z) + 1/3*Div(dy(u), dy(v), dy(w_z))) + (nu*ln_rho0_z + nu_z) * Syz) "
        self.viscous_term_w = " (nu*(Lap(w, w_z) + 1/3*Div(  u_z, v_z, dz(w_z)))   + (nu*ln_rho0_z + nu_z) * Szz) "

        self.problem.substitutions['L_visc_u'] = self.viscous_term_u
        self.problem.substitutions['L_visc_v'] = self.viscous_term_v
        self.problem.substitutions['L_visc_w'] = self.viscous_term_w
        
        self.nonlinear_viscous_u = " nu*(dx(ln_rho1)*Sxx + dy(ln_rho1)*Sxy + dz(ln_rho1)*Sxz) "
        self.nonlinear_viscous_v = " nu*(dx(ln_rho1)*Sxy + dy(ln_rho1)*Syy + dz(ln_rho1)*Syz) "
        self.nonlinear_viscous_w = " nu*(dx(ln_rho1)*Sxz + dy(ln_rho1)*Syz + dz(ln_rho1)*Szz) "
 
        self.problem.substitutions['R_visc_u'] = self.nonlinear_viscous_u
        self.problem.substitutions['R_visc_v'] = self.nonlinear_viscous_v
        self.problem.substitutions['R_visc_w'] = self.nonlinear_viscous_w

        # double check implementation of variabile chi and background coupling term.
        self.linear_thermal_diff      = " ( Cv_inv*( chi*(Lap(T1, T1_z) + T0_z*dz(ln_rho1) + T1_z*ln_rho0_z) + chi_z*T1_z ) )"
        self.nonlinear_thermal_diff   = " ( Cv_inv*chi*(dx(T1)*dx(ln_rho1) + dy(T1)*dy(ln_rho1) + T1_z*dz(ln_rho1)))"

        self.problem.substitutions['L_thermal']    = self.linear_thermal_diff
        self.problem.substitutions['R_thermal']    = self.nonlinear_thermal_diff
        self.problem.parameters['source_terms'] = 0
#        self.source                   = " (Cv_inv*( chi*(T0_zz + ln_rho0_z*T0_z) + chi_z*T0_z) )"
#        self.problem.substitutions['source_terms'] = self.source

        if viscous_heating:
            self.problem.substitutions['R_visc_heat'] = " (Cv_inv*nu*(dx(u)*Sxx + dy(v)*Syy + w_z*Szz + Sxy**2 + Sxz**2 + Syz**2))"
        else:
            self.problem.parameters['R_visc_heat'] = 0

    def _setup_diffusivities(self, Reynolds, Prandtl, epsilon, radius, **kwargs):
        """
        Setup the diffusivities based on a few input values. Chi and Nu are constants.

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
        L = 2*radius
        self.problem.parameters['L'] = L
        nu_top    = np.sqrt(L**3*epsilon*self.atmosphere.params['g']/self.atmosphere.params['Cp'])/Reynolds
        chi_top = nu_top/Prandtl
        
        self.problem.parameters['chi'] = chi_top
        self.problem.substitutions['chi_z'] = '0'
        self.problem.parameters['nu'] = nu_top
        self.problem.substitutions['nu_z'] = '0'
