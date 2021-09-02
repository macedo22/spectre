# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def phi(spatial_metric):
    return pow(np.linalg.det(spatial_metric), -1.0 / 6)


def phi_squared(phi):
    return phi * phi


def conformal_spatial_metric(phi_squared, spatial_metric):
    return phi_squared * spatial_metric


def trace_extrinsic_curvature(extrinsic_curvature, inverse_spatial_metric):
    return np.einsum("ij,ij", extrinsic_curvature, inverse_spatial_metric)


def a_tilde(phi_squared, extrinsic_curvature, inverse_spatial_metric,
            spatial_metric):
    return (phi_squared * (np.einsum("ij", extrinsic_curvature) -
                           (1.0 / 3) * trace_extrinsic_curvature(
                               extrinsic_curvature, inverse_spatial_metric) *
                           np.einsum("ij", spatial_metric)))
