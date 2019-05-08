"""
Dedalus script for simulating a 2D, azimuthally-symmetric thermal in an adiabatic polytrope.
Equations are LBR Anelastic, as in Brown et al. 2012 and Lecoanet et al. 2014, rescaled
on a freefall velocity scale with the length scale set by the diameter of the thermal.

This code must be used with a version of Dedalus after commit  0128fa0 by Keaton Burns 
on 2019-02-21, such that the problem.add_equation() function has the 'tau' keyword.

A 'Buoyancy time' is the freefall timescale if the length were set as the domain depth,
rather than the thermal depth. In our nondimensional units, if the domain depth is
Lz, t_b = sqrt(Lz), and in general it takes ~12 buoyancy times for a "boussinesq" thermal
to cross the domain (e.g., n_rho = 0.01), whereas more stratified thermals take less time
(e.g., at n_rho = 2, it takes about 7 buoyancy times to cross the domain).

Usage:
    AN_2D_thermal.py [options]

Options:
    --n_rho=<n_rho>       number of density scale heights [default: 0.5]
    --Reynolds=<Re>       Reynolds number [default: 6e2]
    --Prandtl=<Pr>        Prandtl number [default: 1]
    --nz=<n>              Number of z coefficients [default: 256]
    --nr=<n>              Number of r coefficients [default: 128]

    --wall_hours=<t>      Number of wall hours to run for max [default: 7.5]
    --run_time_buoy=<rt>  Number of buoyancy times to run for [default: 10.0]

    --aspect=<a>          Domain runs from [0, Lr], with Lr = aspect*Lz [default: 0.25]
    --label=<l>           an optional string at the end of the output directory for easy identification

    --rk443               If flagged, timestep using RK443. Else, SBDF2.

    --Lz=<L>              The depth of the domain, in thermal diameters [default: 20]

    --chi_nu              If true, use a constant diffusivity (rather than constant dynamic diffusivity) eqn formulation
    --safety=<s>          Safety factor base [default: 0.08]
    --out_cadence=<o>     Time cadence of output saves in buoyancy times [default: 0.08]  

    --restart=<file>      Name of file to restart from, if starting from checkpoint

"""
import os
import time
import sys

import numpy as np
from mpi4py import MPI
from scipy.special import erf
from dedalus import public as de
from dedalus.tools  import post
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

from docopt import docopt
args = docopt(__doc__)

from logic.checkpointing import Checkpoint

################
# Read in args
################
n_rho       = float(args['--n_rho'])
Re          = float(args['--Reynolds'])
Pr          = float(args['--Prandtl'])
nz          = int(args['--nz'])
nr          = int(args['--nr'])
kappa_mu    = not(args['--chi_nu'])
aspect      = float(args['--aspect'])
Lz          = float(args['--Lz'])

################
# Set up atmosphere info
################
gamma     = 5./3                         #adiabatic index
m_ad      = 1/(gamma-1)
Cp        = gamma*m_ad
Cv        = Cp/gamma
grad_T_ad = -(np.exp(n_rho/m_ad) - 1)/Lz #adiabatic temperature gradient
g         = (1 + m_ad) #* -grad_T_ad     #gravity
t_b       = np.sqrt(Lz)                  #atmospheric buoyancy time

Lr        = aspect*Lz
radius    = 0.5       #Thermal radius, by definition, in nondimensionalization
delta_r   = radius/5  #Thermal smoothing width

#(r0, z0) is the midpoint of the (spherical) thermal
r0        = 0
z0        = Lz - 3*radius

# Adjust the Re at the top of the domain if in a kappa_mu formulation so that the input Re 
# is the freefall Re at the height of the initial thermal
if kappa_mu:
    Re /= (1 + grad_T_ad*(z0 - Lz))**m_ad

logger.info('Creating polytrope with grad_T_ad: {:.4e}, g: {:.4e}'.format(grad_T_ad, g))

###################
# Set up output dir
###################
data_dir = './'+sys.argv[0].split('.py')[0]
data_dir += '_nrho{:s}_Re{:s}_Pr{:s}_aspect{:s}_Lz{:s}'.format(args['--n_rho'], args['--Reynolds'], args['--Prandtl'], args['--aspect'], args['--Lz'])
if not kappa_mu:
    data_dir += '_ChiNu'
if args['--label'] is not None:
    data_dir += '_{:s}'.format(args['--label'])
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('{:s}/'.format(data_dir)):
        os.mkdir('{:s}/'.format(data_dir))
logger.info('saving files in {:s}'.format(data_dir))

#####################
# Dedalus simulation
#####################
z_basis = de.Fourier('z', nz, interval=(0, Lz), dealias=3/2)
r_basis = de.Chebyshev('r', nr, interval=(0, Lr), dealias=3/2)
domain = de.Domain([z_basis, r_basis], grid_dtype=np.float64)

problem = de.IVP(domain, variables=['u', 'w', 'ur', 'wr', 'S1', 'S1r', 'p'])
problem.meta['u', 'w', 'S1r', 'p']['r']['dirichlet'] = True
problem.parameters['u_phi'] = problem.parameters['u_phir'] = 0

problem.parameters['pi']         = np.pi
problem.parameters['Lz']         = Lz
problem.parameters['Lr']         = Lr
problem.parameters['m_ad']       = m_ad
problem.parameters['Cp']         = Cp
problem.parameters['Cv']         = Cv
problem.parameters['grad_T_ad']  = grad_T_ad
problem.parameters['g']          = g
problem.parameters['Re']         = Re
problem.parameters['Pr']         = Pr
problem.parameters['Lnondim']    = -grad_T_ad #scaling factor from unscaled polytropes -- for viscous heating term.

problem.substitutions['T']            = '(1 + grad_T_ad*(z - Lz))'
problem.substitutions['T0']           = '(T)'
problem.substitutions['rho0']         = '((T)**(m_ad))'              
problem.substitutions['ln_rho0_z']    = '(m_ad*grad_T_ad/T)'         
problem.substitutions['ln_rho0_zz']   = '(-m_ad*grad_T_ad**2/T**2)'  
problem.substitutions['ln_T0_z']      = '(grad_T_ad/T)'              

#Set up the diffusivity profile if nonconstant
if kappa_mu:
    problem.substitutions['xi']     = '(1/(T**m_ad))'
    problem.substitutions['xi_L']   =  '1'
    problem.substitutions['xi_R']   = '((xi - xi_L))'
else:
    problem.parameters['xi']        = 1
    problem.parameters['xi_L']      = 1
    problem.parameters['xi_R']      = 0

#Operators (need some fancy ones in cylindrical coordinates
problem.substitutions['UdotGrad(A, Ar)']    = '(u*Ar + w*dz(A))'
problem.substitutions['UdotGradU_r']        = '(UdotGrad(u, ur) - u_phi**2/r)'
problem.substitutions['UdotGradU_phi']      = '(UdotGrad(u_phi, u_phir) - u_phi*u/r)'
problem.substitutions['Lap(A, Ar)']         = '(Ar/r + dr(Ar) + dz(dz(A)))'
problem.substitutions['Lap_phi(A, Ar)']     = '(Lap(A, Ar) - A/r**2)'
problem.substitutions['Lap_r(A, Ar)']       = '(Lap(A, Ar) - A/r**2)'
problem.substitutions['DivU_true']          = '(u/r + ur + dz(w))'
problem.substitutions['DivU']               = '(DivU_true)'
problem.substitutions['DivUr']              = '(ur/r - u/r**2 + dr(ur) + dz(wr))'
problem.substitutions['DivUz']              = '(-1)*(ln_rho0_z*dz(w) + w*ln_rho0_zz)'

#Stress Tensor Components
problem.substitutions['t_rr']               = '(2*ur      - (2/3)*DivU)'
problem.substitutions['t_zz']               = '(2*dz(w)   - (2/3)*DivU)'
problem.substitutions['t_phi2']             = '(2*u/r     - (2/3)*DivU)'
problem.substitutions['t_rz']               = '(wr + dz(u))'
problem.substitutions['t_phiz']             = '(dz(u_phi))'
problem.substitutions['t_phir']             = '(u_phir - u_phi/r)'

problem.substitutions['t_rr_dr']            = '(2*dr(ur)  - (2/3)*DivUr)'
problem.substitutions['t_rz_dr']            = '(dr(wr) + dz(ur))'
problem.substitutions['t_phir_dr']          = '(dr(u_phir) - u_phir/r)'
problem.substitutions['t_zz_dz_L']          = '(2*dz(dz(w)) - (2./3)*dz(DivU))'
problem.substitutions['t_zz_dz_R']          = '(2*dz(dz(w)) - (2./3)*(DivUz))'

#Momentum equation diffusivitiy substitutions
problem.substitutions['visc_L_u']      = '((xi_L/Re)*(Lap_r(u, ur) + (1./3)*DivUr))'
problem.substitutions['visc_L_w']      = '((xi_L/Re)*(Lap(w, wr)   + (1./3)*dz(DivU)))'
problem.substitutions['visc_R_u']      = '((xi_R/Re)*(Lap_r(u, ur) + (1./3)*DivUr))'
problem.substitutions['visc_R_w']      = '((xi_R/Re)*(Lap(w, wr)   + (1./3)*DivUz))'


#Energy equation diffusivity substitutions
problem.substitutions['diff_L']        = '( (xi_L/(Re*Pr*Cp))*Lap(S1, S1r) )'
if kappa_mu:
    problem.substitutions['diff_R']    = '( ((Re*Pr*Cp)**(-1))*(xi*ln_T0_z*dz(S1) + xi_R*Lap(S1, S1r)) )'
else:
    problem.substitutions['diff_R']    = '( xi*((Re*Pr*Cp)**(-1))*(ln_T0_z+ln_rho0_z)*dz(S1) )'
problem.substitutions['visc_heat']     = '(g*Lnondim*xi*(((Cp*Re*T0)**(-1))*(t_rr*dr(u) + t_rz*dz(u) + t_phir*u_phir + t_phiz*dz(u_phi) + t_rz*dr(w) + t_zz*dz(w))))'

#Equation scaling, vorticity substitution
problem.substitutions['scale_m']       = '(r**2)'
problem.substitutions['scale_e']       = '(r)'
problem.substitutions['scale_c']       = '(r)'
problem.substitutions['V']             = '(dz(u) - wr)'
problem.substitutions['Vr']            = '(dz(ur) - dr(wr))'

#Defn
problem.add_equation("(S1r - dr(S1)) = 0")
problem.add_equation("(ur  - dr(u))  = 0")
problem.add_equation("(wr  - dr(w))  = 0")
#Continuity
problem.add_equation("scale_c*(DivU_true) = -scale_c*(w*ln_rho0_z)", tau=False)
#Momentum equations
problem.add_equation("scale_m*(dt(u) +  dr(p)      - visc_L_u) = scale_m*(-UdotGradU_r     + visc_R_u    )", tau=False)
problem.add_equation("scale_m*(dt(w) +  dz(p) - S1 - visc_L_w) = scale_m*(-UdotGrad(w, wr) + visc_R_w    )", tau=False)
#Entropy equation
problem.add_equation("scale_e*(dt(S1)              - diff_L )  = scale_e*(-UdotGrad(S1, S1r) + visc_heat + diff_R)", tau=False)

#For whatever voodoo reason, these are the only BCs 
#I've found that are stable at high stratification.
problem.add_bc("right(V)   = 0", condition="nz != 0") 
problem.add_bc("right(p)   = 0", condition="nz == 0") 
problem.add_bc("right(w)   = 0") 
problem.add_bc("right(S1r) = 0")
#problem.add_bc("right(u)   = 0") 

#########################
# Initialization of run
#########################
if args['--rk443']:
    solver = problem.build_solver(de.timesteppers.RK443)
else:
    solver = problem.build_solver(de.timesteppers.SBDF2)
logger.info('Solver built')

# Set up initial conditions
r = domain.grid(1)
z = domain.grid(0)
S1 = solver.state['S1']
S1r = solver.state['S1r']

logger.info('checkpointing in {}'.format(data_dir))
checkpoint = Checkpoint(data_dir)
mode = 'overwrite'
restart = args['--restart']
#initial conditions
if restart is None:
    r_IC = np.sqrt((z - z0)**2 + (r - r0)**2)
    S1['g'] = -1*(1 - erf((r_IC - radius)/delta_r))/2
    S1.differentiate('r', out=S1r)
    # Initial timestep
    start_dt = 1e-3*t_b
else:
    logger.info("restarting from {}".format(restart))
    start_dt = checkpoint.restart(restart, solver)
    mode = 'append'
checkpoint.set_checkpoint(solver, sim_dt=1, mode=mode)

# Simulation termination parameters
solver.stop_sim_time = t_b * float(args['--run_time_buoy']) + solver.sim_time
solver.stop_wall_time = 60 * 60. * float(args['--wall_hours'])
solver.stop_iteration = np.inf

# Analysis & outputs
out_dt = float(args['--out_cadence'])*t_b
slices   = solver.evaluator.add_file_handler('{:s}/slices'.format(data_dir),   sim_dt=out_dt, max_writes=20, mode='overwrite')
profiles = solver.evaluator.add_file_handler('{:s}/profiles'.format(data_dir), sim_dt=out_dt, max_writes=20, mode='overwrite')
scalars  = solver.evaluator.add_file_handler('{:s}/scalars'.format(data_dir),   sim_dt=out_dt, max_writes=1e4, mode='overwrite')
for f in ['S1', 'S1r', 'u', 'ur', 'w', 'wr', 'V', 'Vr', 'p']:
    slices.add_task(f, name=f, layout='g')
    profiles.add_task('integ({}, "z")/Lz'.format(f), name='{}_z_avg'.format(f), layout='g')
    profiles.add_task('integ(r*({}), "r")/Lr'.format(f), name='{}_r_avg'.format(f), layout='g')
    scalars.add_task( 'integ(r*{})/Lz/Lr'.format(f), name='{}'.format(f), layout='g')

for f, nm in [('-S1r', 'circ_buoy'), ('V', 'circulation')]:
    scalars.add_task('integ({})'.format(f), name='{}'.format(nm), layout='g')

for f, nm in [('(rho0*w**2)', 'KE_w'), ('(rho0*u**2)', 'KE_u'), ('rho0*S1', 'tot_entropy')]:
    scalars.add_task('integ(2*pi*r*{})'.format(f), name='{}'.format(nm), layout='g')

# CFL
safety_factor = float(args['--safety'])
if args['--rk443']:
    safety_factor *= 4
CFL = flow_tools.CFL(solver, initial_dt=start_dt, cadence=1, safety=safety_factor,
                     max_change=1.5, min_change=0.5, max_dt=1e-3*t_b, threshold=0.05)
CFL.add_velocities(('w', 'u'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property("sqrt(u**2 + u_phi**2 + w**2)", name='v_rms')
flow.add_property("sqrt(u**2 + u_phi**2 + w**2)*Re/xi", name='Re')
flow.add_property("integ(r*rho0*S1)", name='tot_entropy')
flow.add_property("integ(V)", name='circ')

dt = start_dt
# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 1 == 0:
            logger.info('Iteration: {:.2e}, Time/t_b: {:.2e}, dt/t_b: {:.2e}'.format(solver.iteration, solver.sim_time/t_b, dt/t_b) +\
                        ' Max Re = {:.2e}, Circ = {:.2e}, tot_e = {:.2e}'.format(flow.max('Re'), flow.max('circ'), flow.max('tot_entropy')))
        if np.isnan(flow.max('v_rms')):
            logger.info('NaN, breaking.')
            break
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    final_checkpoint = Checkpoint(data_dir, checkpoint_name='final_checkpoint')
    final_checkpoint.set_checkpoint(solver, wall_dt=1, mode="append")
    solver.step(dt) #clean this up in the future...works for now.
    for t in [checkpoint, final_checkpoint]:
        post.merge_process_files(t.checkpoint_dir, cleanup=False)
    for t in [slices, profiles, scalars]:
        post.merge_process_files(t.base_path, cleanup=False)
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Iter/sec: %.2f ' %(solver.iteration/(end_time-start_time)))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
