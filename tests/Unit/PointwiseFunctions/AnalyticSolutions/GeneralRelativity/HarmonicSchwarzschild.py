# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def harmonic_condition_445_lhs(shift, d_shift, dt_shift):
    return dt_shift - np.einsum("j,ji", shift, d_shift)


def harmonic_condition_445_rhs(lapse, d_ln_lapse, inverse_spatial_metric,
                               christoffel_second_kind):
    return -(lapse)**2 * (
        np.einsum("ij,j", inverse_spatial_metric, d_ln_lapse) +
        np.einsum("jk,ijk", inverse_spatial_metric, christoffel_second_kind))
