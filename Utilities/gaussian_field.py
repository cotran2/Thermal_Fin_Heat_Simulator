import numpy as np
from scipy import linalg, spatial
from fenics import Function, plot
import matplotlib.pyplot as plt

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def make_cov_chol(V, generate_2D, generate_3D, kern_type='m52', length=1.6):
    V0_dofs = V.dofmap().dofs()
    if generate_2D == 1:
        Wdofs_x = V.tabulate_dof_coordinates().reshape((-1, 2))
        points = Wdofs_x[V0_dofs, :] 
    if generate_3D == 1:
        Wdofs_x = V.tabulate_dof_coordinates().reshape((-1, 3))
        points = Wdofs_x[V0_dofs, :]   
    dists = spatial.distance.pdist(points)
    dists = spatial.distance.squareform(dists)
    
    if kern_type=='sq_exp':
        # Squared Exponential / Radial Basis Function
        alpha = 1 / (2 * length ** 2)
        noise_var = 1e-2
        cov = np.exp(-alpha * dists ** 2) + np.eye(len(points)) * noise_var
    elif kern_type=='m52':
        # Matern52
        tmp = np.sqrt(5) * dists / length
        cov = (1 + tmp + tmp * tmp / 3) * np.exp(-tmp)
    else:
        # Matern32
        tmp = np.sqrt(3) * dists / length
        cov = (1 + tmp) * np.exp(-tmp)
    
    machine_eps = np.finfo(float).eps
    diagonal_perturbation = 10**10*machine_eps*np.identity(cov.shape[0])
    chol = linalg.cholesky(cov + diagonal_perturbation)
    return chol

#  V = get_space(40)
#  chol = make_cov_chol(V)
#  norm = np.random.randn(len(chol))
#  q = Function(V)
#  q.vector().set_local(np.exp(0.5 * chol.T @ norm))

#  f = plot(q)
#  plt.colorbar(f)
#  plt.show()
#  plt.savefig('fin.png')
#  f.write_png('fin.png')
