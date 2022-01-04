# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

# CHECKS THE DERIVATIVE OF FUNCTION F WITH FINITE DIFFERENCE OF SOME
# SMALL PERTUBATION OF F

def combine_pert_coords(x_plus_dx_i_KS, x_plus_dy_i_KS, x_plus_dz_i_KS):
    pertubation_coords = []
    pertubation_coords.append(x_plus_dx_i_KS)
    pertubation_coords.append(x_plus_dy_i_KS)
    pertubation_coords.append(x_plus_dz_i_KS)

    pertubation_coords = np.array(pertubation_coords)
    return pertubation_coords


def check_finite_difference(input_vector, perturbed_input_vectors,
                            pertubation):
    # parameters:
    # input_vector: np.array of size(# of param of f): The function of interest
    # evaluated at the coordinates of interest. Can also be one of
    # the vectors that make up a higher dimensional tensor.

    # perturbed_input_vectors: rank 2 Tensor expressed as a np.array of
    # dim(# of param of f)**2: The function of interest evaluated at the
    # coordinates of interest plus some small pertubation. Need a vector
    # for a pertubation in each direction, so it is a rank two tensor.
    # Can also be a tensor composed of vectors of a small pertubation that
    # make up a higher dimensional perturbed tensor.

    # pertubation: np.array of size(# of param of f): Size of the
    # pertubation for the parameters of the function whose derivative is
    # being taken (for f(x,y,z), dx=dy=dz).

    input_vector = input_vector.tolist()
    perturbed_input_vectors = perturbed_input_vectors.tolist()
    pertubation = pertubation.tolist()
    derivative_tensor = []
    for i in range(len(input_vector)):
        dimension_1 = []
        for j in range(len(input_vector)):
            derivative_tensor_indexed_value = (
                perturbed_input_vectors[j][i] -
                input_vector[i]) / pertubation[i]
            dimension_1.append(derivative_tensor_indexed_value)
        derivative_tensor.append(dimension_1)
    derivative_tensor = np.array(derivative_tensor)
    return derivative_tensor
