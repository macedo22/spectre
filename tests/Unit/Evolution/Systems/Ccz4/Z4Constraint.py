# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def spatial_z4_constraint(conformal_spatial_metric,
                          contracted_conformal_christoffel_second_kind,
                          gamma_hat):
    return (0.5 *
            np.einsum("ij,j", conformal_spatial_metric, gamma_hat -
                      contracted_conformal_christoffel_second_kind))
