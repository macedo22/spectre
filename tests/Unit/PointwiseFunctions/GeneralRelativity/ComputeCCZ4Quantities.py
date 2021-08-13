# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def phi(spatial_metric):
    return pow(np.linalg.det(spatial_metric), -1.0 / 6)


def phi_squared(phi):
    return phi * phi


def conformal_spatial_metric(phi_squared, spatial_metric):
    return phi_squared * spatial_metric
