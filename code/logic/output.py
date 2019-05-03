import numpy as np
from collections import OrderedDict

import logging
logger = logging.getLogger(__name__)

from dedalus import public as de

def initialize_output(data_dir, solver, threeD=False, output_cadence=1, writes_per=2, mode='overwrite', threeD_factor=1):
    """
    Initializes output tasks for a convection simulation.

    Parameters
    ----------
    data_dir    : string
        The root directory in which to save the output tasks
    solver      : A Dedalus solver object
        The solver for the simulation that output is for.
    threeD      : bool, optional
        If True, simulation is 3D.
    output_cadence : float, optional
        The output cadence of writes (in simulation time units)
    writes_per     : int, optional
        The maximum number of output writes per file.
    mode           : string, optional
        "overwrite" or "append" -- The Dedalus output mode.
    threeD_factor  : float, optional
        A factor to multiply output_cadence by to increase (or decrease) the time between outputs.
    """
    # VOLUMES AND SLICES
    analysis_tasks = OrderedDict()
    slices = solver.evaluator.add_file_handler(data_dir+'slices', sim_dt=output_cadence, max_writes=writes_per, mode=mode, parallel=False)
    analysis_tasks['slices'] = slices
    if threeD:
        volumes = solver.evaluator.add_file_handler(data_dir+'volumes', sim_dt=output_cadence*threeD_factor, max_writes=writes_per, mode=mode, parallel=False)
        analysis_tasks['volumes'] = volumes
        volumes.add_task('u',          name='u')
        volumes.add_task('v',          name='v')
        volumes.add_task('w',          name='w')
        volumes.add_task('T1',         name='T1')
        volumes.add_task('s1',         name='entropy')

        slices.add_task('interp(u, x = 0)',         name='u x mid')
        slices.add_task('interp(u, y = 0)',         name='u y mid')
        slices.add_task('interp(v, x = 0)',         name='v x mid')
        slices.add_task('interp(v, y = 0)',         name='v y mid')
        slices.add_task('interp(w, x = 0)',         name='w x mid')
        slices.add_task('interp(w, y = 0)',         name='w y mid')
        slices.add_task('interp(s1, x = 0)',        name='s x mid')
        slices.add_task('interp(s1, y = 0)',        name='s y mid')
        slices.add_task('interp(Vort_x, x = 0)',    name='vorticity x mid')
        slices.add_task('interp(Vort_y, y = 0)',    name='vorticity y mid')
    else:
        slices.add_task('Vort_y',       name='V')
        slices.add_task('u',            name='u')
        slices.add_task('w',            name='w')
        slices.add_task('T1',           name='T1')
        slices.add_task('rho_fluc',     name='rho_fluc')
        slices.add_task('s1',           name='S1')
        slices.add_task('enstrophy',    name='enstrophy')

    #PROFILES
    profiles = solver.evaluator.add_file_handler(data_dir+'profiles', sim_dt=output_cadence, max_writes=writes_per, mode=mode, parallel=False)
    analysis_tasks['profiles'] = profiles
    profiles.add_task("plane_avg(T_full)",          name="T_full")
    profiles.add_task("plane_avg(rho_full)",        name="rho_full")
    profiles.add_task("plane_avg(rho_fluc)",        name="rho_fluc")
    profiles.add_task("plane_avg(s1)",              name="s_fluc")
    profiles.add_task("plane_avg(rho_full*s1)",     name="s_full_fluc")
    profiles.add_task("plane_avg(w)",               name="w")
    profiles.add_task("plane_avg(vel_rms)",         name="vel_rms")
    profiles.add_task("plane_avg(Re_rms)",    name="Re_rms")
    profiles.add_task("plane_avg(Pe_rms)",   name="Pe_rms")
    profiles.add_task("plane_avg(Ma_rms)",          name="Ma_rms")
    profiles.add_task("plane_avg(T0)",              name="T0")
    profiles.add_task("plane_avg(rho0)",            name="rho0")
    profiles.add_task("plane_avg(enstrophy)",       name="enstrophy")

    #SCALARS
    scalars = solver.evaluator.add_file_handler(data_dir+'scalars', sim_dt=output_cadence, max_writes=writes_per, mode=mode, parallel=False)
    analysis_tasks['scalars'] = scalars
    scalars.add_task("vol_avg(rho_full*Cv*T1 + rho_fluc*Cv*T0)",            name='IE_fluc')
    scalars.add_task("vol_avg(rho_fluc*phi)",                               name='PE_fluc')
    scalars.add_task("vol_avg(rho_full*(vel_rms**2)/2)",                    name='KE')
    scalars.add_task("integ(s1)",                                           name="s1_integ") #technically s1/cp.
    scalars.add_task("integ(rho_full*s1)",                                  name="rho_s1_integ")
    scalars.add_task("integ(Cv*(L_thermal + R_thermal)/(T_full))",          name="s1_diff")
    scalars.add_task("integ(rho_full*Cv*(L_thermal + R_thermal)/(T_full))", name="rho_s1_diff")
    scalars.add_task("integ(Cv*R_visc_heat/(T_full))",                      name="s1_VH")
    scalars.add_task("integ(rho_full*Cv*R_visc_heat/(T_full))",             name="rho_s1_VH")

    return analysis_tasks
