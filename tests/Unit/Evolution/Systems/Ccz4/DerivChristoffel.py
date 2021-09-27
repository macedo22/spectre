# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def deriv_conformal_christoffel_second_kind(inverse_conformal_spatial_metric,
                                            field_d, d_field_d, field_d_up):
    return (
        -2.0 * np.einsum("kml,ijl->kmij", field_d_up,
                         (np.einsum("ijl", field_d) + np.einsum(
                             "jil", field_d) - np.einsum("lij", field_d))) +
        np.einsum(
            "ml,ijkl->kmij", inverse_conformal_spatial_metric,
            (np.einsum("kijl", d_field_d) + np.einsum("ikjl", d_field_d) +
             np.einsum("kjil", d_field_d) + np.einsum("jkil", d_field_d) -
             np.einsum("klij", d_field_d) - np.einsum("lkij", d_field_d))) /
        2.0)


def deriv_christoffel_second_kind(d_conformal_christoffel_second_kind,
                                  conformal_spatial_metric,
                                  inverse_conformal_spatial_metric, field_d,
                                  field_d_up, field_p, d_field_p):
    return (
        d_conformal_christoffel_second_kind + 2.0 *
        (np.einsum("kml,ijl->kmij", field_d_up,
                   (np.einsum("jl,i", conformal_spatial_metric, field_p) +
                    np.einsum("il,j", conformal_spatial_metric, field_p) -
                    np.einsum("ij,l", conformal_spatial_metric, field_p))) -
         np.einsum("ml,ijkl->kmij", inverse_conformal_spatial_metric,
                   (np.einsum("kjl,i", field_d, field_p) +
                    np.einsum("kil,j", field_d, field_p) -
                    np.einsum("kij,l", field_d, field_p)))) -
        np.einsum("ml,ijkl->kmij", inverse_conformal_spatial_metric,
                  (np.einsum("jl,ki", conformal_spatial_metric, d_field_p) +
                   np.einsum("jl,ik", conformal_spatial_metric, d_field_p) +
                   np.einsum("il,kj", conformal_spatial_metric, d_field_p) +
                   np.einsum("il,jk", conformal_spatial_metric, d_field_p) -
                   np.einsum("ij,kl", conformal_spatial_metric, d_field_p) -
                   np.einsum("ij,lk", conformal_spatial_metric, d_field_p))) /
        2.0)
