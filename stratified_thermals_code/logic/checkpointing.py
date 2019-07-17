import pathlib 
import h5py
import numpy as np
import logging
import re
logger = logging.getLogger(__name__.split('.')[-1])

class Checkpoint:
    """Simple checkpointing."""
    def __init__(self, data_dir, checkpoint_name="checkpoint", excluded_dirs=[], layout = 'c'):
        """Initialize checkpoint save file.  
        

        Parameters
        ----------
        data_dir : str
            Base directory for storing checkpoints.
        solver : dedalus object (dedalus2/pde)
            Dedalus solver object for problem to checkpoint.
        layout : str, optional
            Space to save checkpoint in.  Values are 'c' (coefficient space) or 'g' (grid space).  Default is 'c'.
        excluded_dirs : list, optional
            If there are directories OTHER than dedalus output directories in the specified data_dir, checkpointing
                will crash on initialization.  This is a full list of directories to be EXCLUDED 
                from checkpointing. 
        """ 

        self.data_dir = pathlib.Path(data_dir)
        self.name = checkpoint_name
        self.excluded_dirs = excluded_dirs
        self.checkpoint_dir = self.data_dir.joinpath(self.name)
        self.layout = layout

        # this should be set via some kind of global option
        self.set_re = re.compile("[\w]*_s([0-9]+)")

    def set_checkpoint(self, solver, wall_dt=np.inf, sim_dt=np.inf, iter=np.inf,
                                     parallel=False, mode="append"):
        """

        Parameters
        ----------
        parallel : logical, optional
            If True, utilize parallel hdf5 output.  If False, do per-core output.  Default is False. 
        wall_dt : float, optional
            Wall time cadence for evaluating tasks (default: infinite)
        sim_dt : float, optional
            Simulation time cadence for evaluating tasks (default: infinite)
        iter : int, optional
            Iteration cadence for evaluating tasks (default: infinite)
        mode : string, optional
            If "overwrite", checkpoints will always write checkpoint file 1.  If
            "append," new checkpoints will be created but old checkpoints will
            not be erased
        """

        self.checkpoint = solver.evaluator.add_file_handler(self.checkpoint_dir,
                                                            wall_dt=wall_dt,
                                                            sim_dt=sim_dt,
                                                            iter=iter,max_writes=1,
                                                            parallel=parallel,
                                                            mode=mode)
        self.checkpoint.add_system(solver.state, layout = self.layout)

    def restart(self, checkpoint_file, solver, cp_record=-1):
        """Restart from checkpoint save file.  

        This file must, at present, be a single unified HDF5 file
        (e.g., if parallel=False on write-out, the data must be joined
        before restart).

        """ 
        logger.info(checkpoint_file)
        f = pathlib.Path(checkpoint_file)
        stem = f.stem
        
        try:
            set_num = int(self.set_re.match(stem).group(1))
        except:
            raise FileNotFoundError("Output filename not as expected.")
            
        write, dt = solver.load_state(checkpoint_file, cp_record)

        return dt
