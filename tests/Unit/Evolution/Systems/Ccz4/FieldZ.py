# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def field_z(conformal_spatial_metric,
            contracted_conformal_christoffel_second_kind, gamma_hat):
    return (0.5 *
            np.einsum("ij,j", conformal_spatial_metric, gamma_hat -
                      contracted_conformal_christoffel_second_kind))


def inverse_field_z(conformal_factor,
                    contracted_conformal_christoffel_second_kind, gamma_hat):
    return 0.5 * conformal_factor**2 * (
        gamma_hat - contracted_conformal_christoffel_second_kind)
