from prody import LOGGER
from prody.utilities import importLA

def fun(x):
    return sqrt(sum(np.array(x) ** 2))

def calcPairDeformationDist2(model, coords, ind1, ind2, kbt=1.):                                       
    """Returns distribution of the deformations in the distance contributed by each mode 
    for selected pair of residues *ind1* *ind2* using *model* from a :class:`.ANM`.
    Method described in [EB08]_ equation (10) and figure (2).     
    
    .. [EB08] Eyal E., Bahar I. Toward a Molecular Understanding of 
        the Anisotropic Response of Proteins to External Forces:
        Insights from Elastic Network Models. *Biophys J* **2008** 94:3424-34355. 
    
    :arg model: this is an 3-dimensional :class:`NMA` instance from a :class:`.ANM`
        calculations.
    :type model: :class:`.ANM`  

    :arg coords: a coordinate set or an object with :meth:`getCoords` method.
        Recommended: ``coords = parsePDB('pdbfile').select('protein and name CA')``.
    :type coords: :class:`~numpy.ndarray`.

    :arg ind1: first residue number.
    :type ind1: int 
    
    :arg ind2: second residue number.
    :type ind2: int 
    """

    try:
        resnum_list = coords.getResnums()
        resnam_list = coords.getResnames()
        coords = (coords._getCoords() if hasattr(coords, '_getCoords') else
                coords.getCoords())
    except AttributeError:
        try:
            checkCoords(coords)
        except TypeError:
            raise TypeError('coords must be a Numpy array or an object '
                            'with `getCoords` method')
    
    if not isinstance(model, NMA):
        raise TypeError('model must be a NMA instance')
    elif not model.is3d():
        raise TypeError('model must be a 3-dimensional NMA instance')
    elif len(model) == 0:
        raise ValueError('model must have normal modes calculated')
    
    linalg = importLA()
    n_atoms = model.numAtoms()
    n_modes = model.numModes()
    LOGGER.timeit('_pairdef')

    r_ij = np.zeros((n_atoms,n_atoms,3))
    r_ij_norm = np.zeros((n_atoms,n_atoms,3))

    for i in range(n_atoms):
        for j in range(i+1,n_atoms):
            r_ij[i][j] = coords[j,:] - coords[i,:]
            r_ij[j][i] = r_ij[i][j]
            r_ij_norm[i][j] = r_ij[i][j]/linalg.norm(r_ij[i][j])
            r_ij_norm[j][i] = r_ij_norm[i][j]

    eigvecs = model.getEigvecs()
    eigvals = model.getEigvals()
    
    D_pair_k = []
    mode_nr = []
    d_vector = []
    
    ind1 = ind1 - resnum_list[0]
    ind2 = ind2 - resnum_list[0]

    for m in range(6,n_modes):
        U_ij_k = [(eigvecs[ind1*3][m] - eigvecs[ind2*3][m]), (eigvecs[ind1*3+1][m] \
            - eigvecs[ind2*3+1][m]), (eigvecs[ind1*3+2][m] - eigvecs[ind2*3+2][m])] 
        D_ij_k = abs(sqrt(kbt/eigvals[m])*(np.vdot(r_ij_norm[ind1][ind2], U_ij_k)))

        # Get the vector d_ij using equation (10) on Biophys J 2008 94:3424-34355.
        # https://doi.org/10.1529/biophysj.107.120733
        kbt_over_lambda = kbt/eigvals[m]
        cos_alpha = np.vdot(r_ij[ind2][ind1], U_ij_k) / (fun(r_ij[ind2][ind1]) * fun(U_ij_k))
        deltaRj = np.array([eigvecs[ind2*3][m], eigvecs[ind2*3+1][m], eigvecs[ind2*3+2][m]])
        deltaRi = np.array([eigvecs[ind1*3][m], eigvecs[ind1*3+1][m], eigvecs[ind1*3+2][m]])
        d_vec = sqrt(kbt_over_lambda) * cos_alpha * (deltaRj - deltaRi)
        
        D_pair_k.append(D_ij_k)
        mode_nr.append(m)
        d_vector.append(d_vec)

    LOGGER.report('Deformation was calculated in %.2lfs.', label='_pairdef')
    
    return mode_nr, D_pair_k, r_ij[ind2][ind1], d_vector
