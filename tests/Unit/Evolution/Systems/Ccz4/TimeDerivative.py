# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def conformal_factor(det_spatial_metric):
    return pow(det_spatial_metric, -1 / 6)


def conformal_spatial_metric(conformal_factor, spatial_metric):
    return conformal_factor * conformal_factor * spatial_metric


def deriv_conformal_spatial_metric(conformal_factor, spatial_metric,
                                   d_spatial_metric, d_det_spatial_metric):
    return (pow(conformal_factor, 2) * d_spatial_metric -
            pow(conformal_factor, 8) *
            np.einsum("k,ij->kij", d_det_spatial_metric, spatial_metric) / 3)


def field_a(d_lapse, lapse):
    return d_lapse / lapse


def field_d(d_conformal_spatial_metric):
    return 0.5 * d_conformal_spatial_metric


def field_p(det_spatial_metric, d_det_spatial_metric):
    return -d_det_spatial_metric / (6 * det_spatial_metric)


def field_d_up(d_inverse_spatial_metric):
    return -d_inverse_spatial_metric


def trace_extrinsic_curvature(extrinsic_curvature, inverse_spatial_metric):
    return np.einsum("ij,ij", extrinsic_curvature, inverse_spatial_metric)
