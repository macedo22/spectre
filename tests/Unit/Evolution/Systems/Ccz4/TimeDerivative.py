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


def inverse_conformal_spatial_metric(conformal_factor, inverse_spatial_metric):
    return inverse_spatial_metric / (conformal_factor * conformal_factor)


# eq 4g solved for theta
def theta():
    return 0  # TODO


# eq 4h solved for b^i
def b():
    return 0  # TODO


# eq 6
def field_a(d_lapse, lapse):
    return d_lapse / lapse


# eq 6
def field_d(d_conformal_spatial_metric):
    return 0.5 * d_conformal_spatial_metric


# eq 6
def field_p(det_spatial_metric, d_det_spatial_metric):
    return -d_det_spatial_metric / (6 * det_spatial_metric)


def trace_extrinsic_curvature(extrinsic_curvature, inverse_spatial_metric):
    return np.einsum("ij,ij", extrinsic_curvature, inverse_spatial_metric)


# eq 3
def a_tilde(conformal_factor_squared, spatial_metric, extrinsic_curvature,
            trace_extrinsic_curvature):
    return conformal_factor_squared * (
        extrinsic_curvature - trace_extrinsic_curvature * spatial_metric / 3.0)


# eq 13
def trace_a_tilde(inverse_conformal_spatial_metric, a_tilde):
    return np.einsum("ij,ij", inverse_conformal_spatial_metric, a_tilde)


# eq 14
def field_d_up(inverse_conformal_spatial_metric, field_d):
    return np.einsum("in,mj,knm", inverse_conformal_spatial_metric,
                     inverse_conformal_spatial_metric, field_d)


# eq 15
def conformal_christoffel_second_kind(inverse_conformal_spatial_metric,
                                      field_d):
    return np.einsum("kl,ijl->kij", inverse_conformal_spatial_metric,
                     (np.einsum("ijl", field_d) + np.einsum("jil", field_d) -
                      np.einsum("lij", field_d)))


# eq 16
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


# eq 17
def christoffel_second_kind(conformal_spatial_metric,
                            inverse_conformal_spatial_metric, field_p,
                            conformal_christoffel_second_kind):
    return (
        np.einsum("kij->kij", conformal_christoffel_second_kind) -
        (np.einsum("kl,ijl->kij", inverse_conformal_spatial_metric,
                   (np.einsum("jl,i", conformal_spatial_metric, field_p) +
                    np.einsum("il,j", conformal_spatial_metric, field_p) -
                    np.einsum("ij,l", conformal_spatial_metric, field_p)))))


# eq 18
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


# eq 19-20
def spatial_ricci_tensor(christoffel_second_kind,
                         d_conformal_christoffel_second_kind,
                         conformal_spatial_metric,
                         inverse_conformal_spatial_metric, field_d, field_d_up,
                         field_p, d_field_p):
    d_christoffel_second_kind = deriv_christoffel_second_kind(
        d_conformal_christoffel_second_kind, conformal_spatial_metric,
        inverse_conformal_spatial_metric, field_d, field_d_up, field_p,
        d_field_p)

    return (
        np.einsum("mmij", d_christoffel_second_kind) -
        np.einsum("jmim", d_christoffel_second_kind) + np.einsum(
            "lij,mlm", christoffel_second_kind, christoffel_second_kind) -
        np.einsum("lim,mlj", christoffel_second_kind, christoffel_second_kind))


# eq 21
def grad_grad_lapse(lapse, christoffel_second_kind, field_a, d_field_a):
    return (lapse * np.einsum("i,j", field_a, field_a) -
            lapse * np.einsum("kij,k", christoffel_second_kind, field_a) +
            0.5 * lapse *
            (np.einsum("ij", d_field_a) + np.einsum("ij->ji", d_field_a)))


# eq 22
def divergence_lapse(conformal_factor_squared,
                     inverse_conformal_spatial_metric, grad_grad_lapse):
    return (
        conformal_factor_squared *
        np.einsum("ij,ij", inverse_conformal_spatial_metric, grad_grad_lapse))


# eq 23
def contracted_conformal_christoffel_second_kind(
    inverse_conformal_spatial_metric, conformal_christoffel_second_kind):
    return np.einsum("jl,ijl", inverse_conformal_spatial_metric,
                     conformal_christoffel_second_kind)


# eq 24
def deriv_contracted_conformal_christoffel_second_kind(
    inverse_conformal_spatial_metric, field_d_up,
    conformal_christoffel_second_kind, d_conformal_christoffel_second_kind):
    return (-2.0 * np.einsum("kjl,ijl->ki", field_d_up,
                             conformal_christoffel_second_kind) +
            np.einsum("jl,kijl->ki", inverse_conformal_spatial_metric,
                      d_conformal_christoffel_second_kind))


# eq 25
def spatial_z4_constraint(conformal_spatial_metric,
                          gamma_hat_minus_contracted_conformal_christoffel):
    return (0.5 * np.einsum("ij,j", conformal_spatial_metric,
                            gamma_hat_minus_contracted_conformal_christoffel))


# eq 25
def upper_spatial_z4_constraint(
    conformal_factor_squared,
    gamma_hat_minus_contracted_conformal_christoffel):
    return (0.5 * conformal_factor_squared *
            gamma_hat_minus_contracted_conformal_christoffel)


# eq 26
def grad_spatial_z4_constraint(
    spatial_z4_constraint, conformal_spatial_metric, christoffel_second_kind,
    field_d, gamma_hat_minus_contracted_conformal_christoffel,
    d_gamma_hat_minus_contracted_conformal_christoffel):
    return (
        np.einsum("ijl,l", field_d,
                  gamma_hat_minus_contracted_conformal_christoffel) +
        0.5 * np.einsum("jl,il", conformal_spatial_metric,
                        d_gamma_hat_minus_contracted_conformal_christoffel) -
        np.einsum("lij,l", christoffel_second_kind, spatial_z4_constraint))


# eq 27
def ricci_scalar_plus_divergence_z4_constraint(
    conformal_factor_squared, inverse_conformal_spatial_metric,
    spatial_ricci_tensor, grad_spatial_z4_constraint):
    return conformal_factor_squared * np.einsum(
        "ij,ij", inverse_conformal_spatial_metric,
        spatial_ricci_tensor + grad_spatial_z4_constraint +
        np.einsum("ji", grad_spatial_z4_constraint))


# eq 12a
def dt_conformal_metric(shift, field_d, conformal_spatial_metric, field_b,
                        lapse, a_tilde, one_over_relaxation_time):
    return (np.einsum("k,kij", shift, field_d) +
            np.einsum("ki,jk", conformal_spatial_metric, field_b) +
            np.einsum("kj,ik", conformal_spatial_metric, field_b) -
            2.0 * np.einsum("ij,kk", conformal_spatial_metric, field_b) / 3.0 -
            2.0 * lapse *
            (a_tilde - conformal_spatial_metric *
             trace_a_tilde(inverse_conformal_spatial_metric, a_tilde) / 3.0) -
            one_over_relaxation_time *
            (np.linalg.det(conformal_spatial_metric) - 1) *
            conformal_spatial_metric)


# eq 12b
def dt_ln_lapse(shift, field_a, lapse, slicing_condition,
                trace_extrinsic_curvature, k_0, theta, c):
    return (np.einsum("k,k", shift, field_a) - lapse * slicing_condition *
            (extrinsic_curvature - k_0 - 2 * theta * c))


# eq 12c
def dt_shift(s, shift, field_b, f, b):
    return s * np.einsum("k,ki", shift, field_b) + s * f * b


# eq 12d
def dt_ln_conformal_factor(shift, field_p, lapse, extrinsic_curvature,
                           field_b):
    return np.einsum("k,k", shift, field_p) + (lapse * extrinsic_curvature -
                                               np.einsum("kk", field_b) / 3.0)


# eq 12e
def dt_a_tilde(lapse, christoffel_second_kind, field_a, d_field_a, gamma_hat,
               contracted_conformal_christoffel, conformal_spatial_metric,
               conformal_factor_squared, inverse_conformal_spatial_metric,
               d_gamma_hat, d_contracted_conformal_christoffel, field_d,
               spatial_ricci_tensor, extrinsic_curvature, a_tilde):
    grad_grad_lapse = grad_grad_lapse(lapse, christoffel_second_kind, field_a,
                                      d_field_a)
    gamma_hat_minus_contracted_conformal_christoffel = (
        gamma_hat - contracted_conformal_christoffel)
    spatial_z4_constraint = spatial_z4_constraint(
        conformal_spatial_metric,
        gamma_hat_minus_contracted_conformal_christoffel)
    divergence_lapse = divergence_lapse(conformal_factor_squared,
                                        inverse_conformal_spatial_metric,
                                        grad_grad_lapse)
    d_gamma_hat_minus_contracted_conformal_christoffel = (
        d_gamma_hat - d_contracted_conformal_christoffel)
    grad_spatial_z4_constraint = grad_spatial_z4_constraint(
        spatial_z4_constraint, conformal_spatial_metric,
        christoffel_second_kind, field_d,
        gamma_hat_minus_contracted_conformal_christoffel,
        d_gamma_hat_minus_contracted_conformal_christoffel)
    ricci_scalar_plus_divergence_z4_constraint = (
        ricci_scalar_plus_divergence_z4_constraint(
            conformal_factor_squared, inverse_conformal_spatial_metric,
            spatial_ricci_tensor, grad_spatial_z4_constraint))
    trace_extrinsic_curvature = trace_a_tilde(inverse_conformal_spatial_metric,
                                              extrinsic_curvature)
    trace_a_tilde = trace_a_tilde(inverse_conformal_spatial_metric, a_tilde)

    return (
        np.einsum("k,kij", shift, d_a_tilde) + conformal_factor_squared *
        (-1.0 * grad_grad_lapse + lapse *
         (spatial_ricci_tensor + grad_spatial_z4_constraint +
          np.einsum("ji->ij", grad_spatial_z4_constraint))) -
        conformal_factor_squared * (conformal_spatial_metric /
                                    (3.0 * conformal_factor_squared)) *
        (-1.0 * np.einsum("kk", divergence_lapse) +
         lapse * ricci_scalar_plus_divergence_z4_constraint) +
        np.einsum("ki,jk", a_tilde, field_b) +
        np.einsum("kj,ik", a_tilde, field_b) -
        2.0 * np.einsum("ij,kk", a_tilde, field_b) / 3.0 + lapse * a_tilde *
        (trace_extrinsic_curvature - 2.0 * theta * c) -
        2.0 * lapse * np.einsum("il,lm,mj", a_tilde,
                                inverse_conformal_spatial_metric, a_tilde) -
        one_over_relaxation_time * conformal_spatial_metric * trace_a_tilde)
