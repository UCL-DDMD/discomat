import numpy as np
from ase import Atoms
from ase.io import write
from ase.build import bulk

# Define standard primitive bases for common crystal structures
PRIMITIVES = {
    'fcc': np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ]),
    'bcc': np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5]
    ]),
    'sc': np.array([
        [0.0, 0.0, 0.0]
    ]),
    'hcp': np.array([
        [0.0, 0.0, 0.0],
        [2/3, 1/3, 0.5]
    ])
}
class CrystalUnitCell:
    def __init__(self, symbol='Cu', crystal_type='fcc', a_lat=None, c_lat=None):
        self.symbol = symbol
        self.crystal_type = crystal_type
        self.bulk_atoms = bulk(symbol, crystal_type, a=a_lat, c=c_lat) # Create bulk structure
        self.lattice_vectors = self.bulk_atoms.cell.array # Get lattice vectors
        # Set the basis based on the crystal type or use the bulk atoms' positions
        if crystal_type in PRIMITIVES:
            self.basis = PRIMITIVES[crystal_type]
        else:
            self.basis = self.bulk_atoms.get_scaled_positions()

    def transform_axes(self, new_axes):
        T = np.array(new_axes).T  # should be 3x3
        inv_T = np.linalg.inv(T)
        self.lattice_vectors = T @ self.lattice_vectors
        self.basis = (inv_T @ self.basis.T).T

    def generate_atoms(self):
        cartesian_positions = self.basis @ self.lattice_vectors
        atoms = Atoms(symbols=[self.symbol] * len(self.basis),
                      positions=cartesian_positions,
                      cell=self.lattice_vectors,
                      pbc=True)
        return atoms

    def save_xyz(self, filename='unitcell.xyz'):
        atoms = self.generate_atoms()
        write(filename, atoms)