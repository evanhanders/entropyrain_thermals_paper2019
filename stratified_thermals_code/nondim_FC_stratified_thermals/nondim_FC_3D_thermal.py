"""
Stratified fully compressible thermals in an adibatically stratified polytrope.
Mu and Kappa are constant with depth.

Usage:
    nondim_FC_3D_thermal.py [options]

Options:
    --Reynolds=<Reynolds>                Reynolds number [default: 5e2]
    --Prandtl=<Prandtl>                  Prandtl number = nu/kappa (top) [default: 1]
    --n_rho=<n_rho>                      Density scale heights across layer [default: 1]
    --epsilon=<epsilon>                  Amplitude of perturbation [default: 1e-4]

    --gamma=<gamma>                      Gamma of ideal gas (cp/cv) [default: 5/3]
    --aspect=<aspect_ratio>              Physical aspect ratio of the atmosphere (Lx = Ly = aspect*[diameter of thermal]) [default: 0.5]

    --nz=<nz>                            vertical z (chebyshev) resolution [default: 512]
    --nx=<nx>                            Horizontal x (Fourier) resolution [default: 256]
    --ny=<ny>                            Horizontal y (Fourier) resolution [default: 256]
    --mesh=<mesh>                        Processor mesh

    --run_time=<run_time>                Run time, in hours [default: 7.5]
    --run_time_buoy=<run_time_buoy>      Run time, in buoyancy times [default: 50]

    --restart=<restart_file>             Restart from checkpoint

    --safety_factor=<safety_factor>      Determines CFL Danger.  Higher=Faster [default: 0.2]

    --label=<label>                      Additional label for run output directory
    --join                               If flagged, join files at end of run
    --out_cadence=<cadence>              Fraction of a buoyancy time on which to output [default: 0.35]                    

    --verbose
    --SBDF2                              If flagged use SBDF2 timestepper
    --443                                If flagged, use RK443 instead of RK222
    --no_VH                              If flagged, don't include viscous heating
"""
import sys
import os
import time
import logging
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
import h5py

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post

from logic.checkpointing import Checkpoint
from logic.output import initialize_output
from logic.domains import DedalusDomain
from logic.atmospheres import Polytrope
from logic.equations import KappaMuFCE
from logic.thermal import Thermal

logger = logging.getLogger(__name__)
checkpoint_min = 30

################################################
# Read in docopt args
################################################

data_dir = './'+sys.argv[0].split('.py')[0]

from docopt import docopt
args = docopt(__doc__)

# Parameters
gamma = float(Fraction(args['--gamma']))
m_ad = 1/(gamma-1)
m = m_ad
n_rho = float(args['--n_rho'])
Lz = 20#np.exp(n_rho/m)-1

#Set thermal properties 
radius = Lz / 40.
z_pert = Lz - 3*radius
delta_r = radius/5 #Sharpness of edge of thermal

aspect_ratio = float(args['--aspect'])
Lx = aspect_ratio*Lz
Ly = aspect_ratio*Lz

threeD = True
restart = args['--restart']

Prandtl  = float(args['--Prandtl'])
Reynolds = float(args['--Reynolds'])
epsilon  = float(args['--epsilon'])
N_buoyancy = args['--run_time_buoy']

if N_buoyancy is not None: N_buoyancy = float(N_buoyancy)

data_dir += '_Re{}_Pr{}_eps{}_nrho{}_aspect{}'.format(args['--Reynolds'], args['--Prandtl'], args['--epsilon'], args['--n_rho'], args['--aspect'])

if args['--no_VH']:
    data_dir += '_noVH'

if args['--label']:
    data_dir += '_{:s}'.format(args['--label'])

data_dir += '/'

logger.info("Saving run in {}".format(data_dir))

import mpi4py.MPI
if mpi4py.MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('{:s}/'.format(data_dir)):
        os.mkdir('{:s}/'.format(data_dir))

nx = int(args['--nx'])
ny = int(args['--ny'])
nz = int(args['--nz'])
mesh = args['--mesh']
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]

#################################################################
# Create atmosphere, equations, and domain
domain = DedalusDomain(nx, ny, nz, Lx, Ly, Lz, threeD=threeD, mesh=mesh)
atmosphere = Polytrope(domain, n_rho=n_rho, aspect_ratio=aspect_ratio, gamma=gamma, R=1, epsilon=epsilon)
buoyancy_time = 1
equations = KappaMuFCE(domain, atmosphere)
Re_adjust = 1/(1 - 3*atmosphere.params['T_ad_z']*radius)**(m_ad) #rho_top / rho_therm
equations.set_operators()
equations.set_equations(Reynolds*Re_adjust, Prandtl, epsilon, radius, viscous_heating=not(args['--no_VH']))
equations.set_BC() #Fixed T

# Build solver
if args['--SBDF2']:
    solver = equations.problem.build_solver(de.timesteppers.SBDF2)
    safety = 1
elif args['--443']:
    solver = equations.problem.build_solver(de.timesteppers.RK443)
    safety = 4
else:
    solver = equations.problem.build_solver(de.timesteppers.RK222)
    safety = 2
logger.info('Solver built')

########################################################
# Timestepping
logger.info("buoyancy_time = {}".format(buoyancy_time))

output_cadence = float(args['--out_cadence'])*buoyancy_time
max_dt = safety * 0.1 * buoyancy_time / 4#np.sqrt(Reynolds)  
dt = max_dt/10

#########################################################
# Initial conditions & Checkpointing
T1      = solver.state['T1']
T1_z    = solver.state['T1_z']
ln_rho1 = solver.state['ln_rho1']

logger.info('checkpointing in {}'.format(data_dir))
checkpoint = Checkpoint(data_dir)
mode = 'overwrite'
if restart is None:
    therm = Thermal(domain, atmosphere, falling=True, radius = radius, r_width=delta_r, A0=epsilon, z_pert=z_pert)
    therm.set_thermal(T1, T1_z, ln_rho1)
else:
    logger.info("restarting from {}".format(restart))
    dt = checkpoint.restart(restart, solver)
    mode = 'append'
checkpoint.set_checkpoint(solver, sim_dt=buoyancy_time, mode=mode)

##############################################
# Outputs and flow tracking

analysis_tasks = initialize_output(data_dir, solver, threeD=threeD, output_cadence=output_cadence, mode=mode)

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property("vol_avg(rho_fluc*phi)", name='PE_fluc')
flow.add_property("Re_rms", name='Re')
flow.add_property("Ma_rms", name='Ma')
flow.add_property("integ(rho_full*s1*Cp)", name='integ_s')
flow.add_property("((dz(T_full)/T_full)**2 - (dz(T0)/T0)**2)", name='dissipation') # good for seeing if the solution isn't resolved.

################################################
# CFL and sim stop conditions
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=safety*float(args['--safety_factor']),
                     max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.05)
CFL.add_velocities(('u', 'v', 'w'))

if N_buoyancy is None:
    solver.stop_sim_time = buoyancy_time*50 + solver.initial_sim_time
else:
    solver.stop_sim_time = N_buoyancy*buoyancy_time + solver.initial_sim_time
solver.stop_wall_time = float(args['--run_time'])*3600
solver.stop_iteration = np.inf + solver.initial_iteration

good_solution = True


Hermitian_cadence = 100
# Main loop
try:
    logger.info('Starting loop, dt {}'.format(dt/buoyancy_time))
    start_time = time.time()
    while solver.ok and good_solution:
        dt = CFL.compute_dt()
        solver.step(dt)

        if (solver.iteration-1) % 1 == 0:
            logger_string = 'iter: {:05d}, t/tb: {:.2e}, dt/tb: {:.2e}'.format(int(solver.iteration), solver.sim_time/buoyancy_time, dt/buoyancy_time)
            Re_avg = flow.grid_average('Re')
            logger_string += ' Integ_s = {:.2e}, PE_fluc = {:.2e}, Max Re = {:.2e}, Max Ma = {:.2e}'.format(flow.max('integ_s'), flow.max('PE_fluc'), flow.max('Re'), flow.max('Ma'))
            logger_string += ', Dissipation min/max: {:.2e} / {:.2e}'.format(flow.min('dissipation'), flow.max('dissipation'))
            logger.info(logger_string)
            if not np.isfinite(Re_avg):
                good_solution = False
                logger.info("Terminating run.  Trapped on Reynolds = {}".format(Re_avg))

        if domain.threeD and solver.iteration % Hermitian_cadence == 0:
            for field in solver.state.fields:
                field.require_grid_space()

        if args['--verbose']:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.spy(solver.pencils[0].L, markersize=1, markeredgewidth=0.0)
            fig.savefig(data_dir+"sparsity_pattern.png", dpi=400)

            import scipy.sparse.linalg as sla
            LU = sla.splu(solver.pencils[0].LHS.tocsc(), permc_spec='NATURAL')
            fig = plt.figure()
            ax = fig.add_subplot(1,2,1)
            ax.spy(LU.L.A, markersize=1, markeredgewidth=0.0)
            ax = fig.add_subplot(1,2,2)
            ax.spy(LU.U.A, markersize=1, markeredgewidth=0.0)
            fig.savefig(data_dir+"sparsity_pattern_LU.png", dpi=400)

            logger.info("{} nonzero entries in LU".format(LU.nnz))
            logger.info("{} nonzero entries in LHS".format(solver.pencils[0].LHS.tocsc().nnz))
            logger.info("{} fill in factor".format(LU.nnz/solver.pencils[0].LHS.tocsc().nnz))

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    final_checkpoint = Checkpoint(data_dir, checkpoint_name='final_checkpoint')
    final_checkpoint.set_checkpoint(solver, wall_dt=1, mode="append")
    solver.step(dt/100) #clean this up in the future...works for now.

    if args['--join']:
        logger.info('beginning join operation')
        logger.info(data_dir+'/final_checkpoint/')   
        post.merge_process_files(data_dir+'/final_checkpoint/', cleanup=False)
        logger.info(data_dir+'/checkpoint/')
        post.merge_process_files(data_dir+'/checkpoint/', cleanup=False)
        for task in analysis_tasks.keys():
            logger.info(analysis_tasks[task].base_path)
            post.merge_process_files(analysis_tasks[task].base_path, cleanup=False)
     
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.domain.dist.comm_cart.size))
    logger.info('Iter/sec: {:g}'.format(solver.iteration/(end_time-start_time)))
