import numpy as np
from collections import OrderedDict

import logging
logger = logging.getLogger(__name__)

from dedalus import public as de

from mpi4py import MPI

class DedalusDomain:
    """
    A struct which contains a dedalus domain, as well as a bunch of info about
    the domain, for use in broad functions.

    Attributes:
    -----------
        nx, ny, nz  : ints
            The coefficient resolution in x, y, z
        Lx, Ly, Lz  : floats
            The size of the domain in simulation units in x, y, z
        threeD      : bool
            If True, problem is 3D. Else, problem is 2D.
        bases       : list
            The dedalus basis objects in the domain
        domain      : a Domain object
            The dedalus domain object
        x, y, z     : NumPy Arrays
            The gridpoints of the domain for x,y,z (scales = 1)
    """

    def __init__(self, nx, ny, nz, Lx, Ly, Lz, threeD=False, mesh=None):
        """
        Initializes the dedalus domain object. Many of the inputs match those in
        the class attribute docstring.

        Other attributes:
        -----------------
        mesh    : list of ints
            The CPU grid on which the domain is distributed.
        """
        self.nx, self.ny, self.nz = nx, ny, nz
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.threeD = threeD

        x_basis = de.Fourier('x', nx, interval=(-Lx/2, Lx/2), dealias=3/2)
        self.bases = [x_basis,]
        if threeD:
            y_basis = de.Fourier('y', ny, interval=(-Ly/2, Ly/2), dealias=3/2)
            self.bases += [y_basis]

        if isinstance(nz, list) and isinstance(Lz, list):
            Lz_int = 0
            z_basis_list = []
            for Lz_i, nz_i in zip(Lz, nz):
                Lz_top = Lz_i + Lz_int
                z_basis = de.Chebyshev('z', nz_i, interval=[Lz_int, Lz_top], dealias=3/2)
                z_basis_list.append(z_basis)
                Lz_int = Lz_top
            self.Lz = Lz_int
            self.nz = np.sum(nz)
            self.Lz_list = Lz
            self.nz_list = nz
            z_basis = de.Compound('z', tuple(z_basis_list), dealias=3/2)
        else:
            z_basis = de.Chebyshev('z', nz, interval=(0, Lz), dealias=3/2)
            self.Lz_list = None
            self.nz_list = None
        self.bases += [z_basis]
        self.domain = de.Domain(self.bases, grid_dtype=np.float64, mesh=mesh, comm=MPI.COMM_WORLD)

        self.x = self.domain.grid(0)
        if threeD:
            self.y = self.domain.grid(1)
        else:
            self.y = None
        self.z = self.domain.grid(-1)

    def new_ncc(self):
        """ Creates a new dedalus field and sets its x & y meta as constant """
        field = self.domain.new_field()
        field.meta['x']['constant'] = True
        if self.threeD:
            field.meta['y']['constant'] = True            
        return field

