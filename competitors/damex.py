import numpy as np
import utilities as ut


#############
# Functions #
#############


def damex_0(x_bin):
    """Calculate the number of points per subfaces from a binary input.

    Parameters:
    x_bin (np.ndarray): Binary input data where rows represent samples and columns represent features.

    Returns:
    tuple: A tuple containing:
        - list: A list of faces where each face is represented by a list of indices of the features.
        - np.ndarray: An array containing the number of samples associated with each face.
    """
    n_sample = x_bin.shape[0]
    n_extr_feats = np.sum(x_bin, axis=1)
    n_shared_feats = np.dot(x_bin, x_bin.T)
    exact_extr_feats = (n_shared_feats == n_extr_feats) * (
        n_shared_feats.T == n_extr_feats).T
    feat_non_covered = set(range(n_sample))
    samples_nb = {}
    for i in range(n_sample):
        feats = list(np.nonzero(exact_extr_feats[i, :])[0])
        if i in list(feat_non_covered):
            feat_non_covered -= set(feats)
            if n_extr_feats[i] > 1:
                samples_nb[i] = len(feats)
    ind_sort = np.argsort(list(samples_nb.values()))[::-1]
    faces = [list(np.nonzero(x_bin[list(samples_nb)[i], :])[0])
             for i in ind_sort]
    mass = [list(samples_nb.values())[i] for i in ind_sort]

    return faces, np.array(mass)


def damex(x_norm, radius, eps, nb_min):
    """Returns faces where the number of points per face is greater than nb_min.

    Parameters:
    x_norm (np.ndarray): Normalized input data from which to generate binary faces.
    radius (float): The radius used to define binary values.
    eps (float): A small value added to avoid numerical issues during calculations.
    nb_min (int): The minimum number of points required for a face to be included in the output.

    Returns:
    list: A list of faces that contain more than nb_min points.
    """
    x_damex = ut.above_radius_bin(x_norm, radius, eps)
    faces, mass = damex_0(x_damex)

    return faces[:np.sum(mass >= nb_min)]


def list_to_dict_size(list_faces):
    """Convert a list of faces to a dictionary where keys are sizes and values are lists of faces.

    Parameters:
    list_faces (list): A list of faces where each face is represented by a list of indices.

    Returns:
    dict: A dictionary where keys are sizes of faces and values are lists of faces of that size.
    """
    faces_dict = {s: [] for s in range(2, max(list(map(len, list_faces)))+1)}
    for face in list_faces:
        faces_dict[len(face)].append(face)

    return faces_dict
