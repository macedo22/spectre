// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/UniformCylindricalFlatEndcapShell.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <pup.h>
#include <sstream>
#include <utility>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Serialization/PupStlCpp11.hpp"

namespace domain::CoordinateMaps {
namespace {

double theta_one(const double shell_fraction, const double theta_inner,
                 const double theta_outer) {
  return theta_inner + shell_fraction * (theta_outer - theta_inner);
}

double radius_two(const double shell_fraction, const double radius_inner,
                  const double radius_outer) {
  return radius_inner + shell_fraction * (radius_outer - radius_inner);
}

double lambda_from_shell_fraction(
    const double shell_fraction, const std::array<double, 3>& center_one,
    const std::array<double, 3>& center_two, const double radius_one,
    const double theta_inner, const double theta_outer,
    const std::array<double, 3>& target_coords) {
  const double r_one_cos_theta =
      radius_one * cos(theta_one(shell_fraction, theta_inner, theta_outer));
  return (target_coords[2] - center_one[2] - r_one_cos_theta) /
         (center_two[2] - center_one[2] - r_one_cos_theta);
}

double function_to_zero(
    const double shell_fraction, const std::array<double, 3>& center_one,
    const std::array<double, 3>& center_two, const double radius_one,
    const double radius_two_inner, const double radius_two_outer,
    const double theta_inner, const double theta_outer,
    const std::array<double, 3>& target_coords) {
  const double theta =
      theta_one(shell_fraction, theta_inner, theta_outer);
  const double lambda = lambda_from_shell_fraction(
      shell_fraction, center_one, center_two, radius_one, theta_inner,
      theta_outer, target_coords);
  const double mapped_radius =
      (1.0 - lambda) * radius_one * sin(theta) +
      lambda * radius_two(shell_fraction, radius_two_inner, radius_two_outer);
  return square(target_coords[0] - center_one[0] -
                lambda * (center_two[0] - center_one[0])) +
         square(target_coords[1] - center_one[1] -
                lambda * (center_two[1] - center_one[1])) -
         square(mapped_radius);
}

}  // namespace

UniformCylindricalFlatEndcapShell::UniformCylindricalFlatEndcapShell(
    const std::array<double, 3>& center_one,
    const std::array<double, 3>& center_two, const double radius_one,
    const double radius_two_inner, const double radius_two_outer,
    const double z_plane_one_inner, const double z_plane_one_outer)
    : center_one_(center_one),
      center_two_(center_two),
      radius_one_(radius_one),
      radius_two_inner_(radius_two_inner),
      radius_two_outer_(radius_two_outer),
      z_plane_one_inner_(z_plane_one_inner),
      z_plane_one_outer_(z_plane_one_outer),
      theta_one_inner_(
          acos((z_plane_one_inner - center_one[2]) / radius_one)),
      theta_one_outer_(
          acos((z_plane_one_outer - center_one[2]) / radius_one)) {
  ASSERT(not equal_within_roundoff(radius_one_, 0.0) and radius_one_ > 0.0,
         "radius_one must be positive");
  ASSERT(not equal_within_roundoff(radius_two_inner_, 0.0) and
             radius_two_inner_ > 0.0,
         "radius_two_inner must be positive");
  ASSERT(radius_two_outer_ > radius_two_inner_,
         "radius_two_outer must be larger than radius_two_inner, but got "
             << radius_two_outer_ << " and " << radius_two_inner_);

  const double cos_theta_inner =
      (z_plane_one_inner_ - center_one_[2]) / radius_one_;
  const double cos_theta_outer =
      (z_plane_one_outer_ - center_one_[2]) / radius_one_;
  ASSERT(abs(cos_theta_inner) < 1.0 and abs(cos_theta_outer) < 1.0,
         "Both planes must intersect sphere one at more than one point. "
         "The cosines of the inner and outer angles are "
             << cos_theta_inner << " and " << cos_theta_outer);
  ASSERT(theta_one_inner_ < theta_one_outer_,
         "The inner spherical angle must be smaller than the outer spherical "
         "angle, but got "
             << theta_one_inner_ << " and " << theta_one_outer_);

#ifdef SPECTRE_DEBUG
  // These bounds match the regime supported by
  // UniformCylindricalFlatEndcap. They are deliberately conservative because
  // the inverse is a one-dimensional numerical root solve.
  const auto param_string = [this]() {
    std::ostringstream buffer;
    buffer << "\nParameters to UniformCylindricalFlatEndcapShell:"
           << "\ncenter_one=" << center_one_ << "\ncenter_two=" << center_two_
           << "\nradius_one=" << radius_one_
           << "\nradius_two_inner=" << radius_two_inner_
           << "\nradius_two_outer=" << radius_two_outer_
           << "\nz_plane_one_inner=" << z_plane_one_inner_
           << "\nz_plane_one_outer=" << z_plane_one_outer_;
    return buffer.str();
  };
  ASSERT(center_two_[2] >= center_one_[2] + 1.05 * radius_one_,
         "center_two[2] must be at least center_one[2] + 1.05 radius_one"
             << param_string());
  ASSERT(center_two_[2] <= center_one_[2] + 5.0 * radius_one_,
         "center_two[2] must be at most center_one[2] + 5 radius_one"
             << param_string());
  ASSERT(theta_one_outer_ < M_PI * 0.35,
         "The outer spherical angle is too large" << param_string());
  ASSERT(theta_one_inner_ > 0.0,
         "The inner spherical angle must be positive" << param_string());
  const double horizontal_center_offset =
      hypot(center_two_[0] - center_one_[0], center_two_[1] - center_one_[1]);
  ASSERT(horizontal_center_offset <=
             radius_one_ * sin(theta_one_inner_),
         "The horizontal distance between centers is too large"
             << param_string());
#endif
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3>
UniformCylindricalFlatEndcapShell::operator()(
    const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& xbar = source_coords[0];
  const ReturnType& ybar = source_coords[1];
  const ReturnType& zbar = source_coords[2];

  const ReturnType rhobar = sqrt(square(xbar) + square(ybar));
  const ReturnType shell_fraction = rhobar - 1.0;
  const ReturnType lambda = 0.5 * (zbar + 1.0);
  const ReturnType theta =
      theta_one_inner_ +
      shell_fraction * (theta_one_outer_ - theta_one_inner_);
  const ReturnType flat_radius =
      radius_two_inner_ +
      shell_fraction * (radius_two_outer_ - radius_two_inner_);
  const ReturnType mapped_radius =
      (1.0 - lambda) * radius_one_ * sin(theta) + lambda * flat_radius;

  return {{center_one_[0] + lambda * (center_two_[0] - center_one_[0]) +
               xbar / rhobar * mapped_radius,
           center_one_[1] + lambda * (center_two_[1] - center_one_[1]) +
               ybar / rhobar * mapped_radius,
           center_one_[2] + lambda * (center_two_[2] - center_one_[2]) +
               (1.0 - lambda) * radius_one_ * cos(theta)}};
}

std::optional<std::array<double, 3>>
UniformCylindricalFlatEndcapShell::inverse(
    const std::array<double, 3>& target_coords) const {
  if ((target_coords[2] < z_plane_one_outer_ and
       not equal_within_roundoff(target_coords[2], z_plane_one_outer_)) or
      (target_coords[2] > center_two_[2] and
       not equal_within_roundoff(target_coords[2], center_two_[2]))) {
    return std::nullopt;
  }

  const double distance_from_center_one =
      sqrt(square(target_coords[0] - center_one_[0]) +
           square(target_coords[1] - center_one_[1]) +
           square(target_coords[2] - center_one_[2]));
  if (distance_from_center_one < radius_one_ and
      not equal_within_roundoff(distance_from_center_one, radius_one_)) {
    return std::nullopt;
  }

  // Reject points outside the ruled surface joining the two outer circles.
  const double outer_lambda =
      (target_coords[2] - z_plane_one_outer_) /
      (center_two_[2] - z_plane_one_outer_);
  const double outer_rho =
      hypot(target_coords[0] - center_one_[0] -
                outer_lambda * (center_two_[0] - center_one_[0]),
            target_coords[1] - center_one_[1] -
                outer_lambda * (center_two_[1] - center_one_[1]));
  const double outer_surface_radius =
      (1.0 - outer_lambda) * radius_one_ * sin(theta_one_outer_) +
      outer_lambda * radius_two_outer_;
  if (outer_rho > outer_surface_radius and
      not equal_within_roundoff(
          outer_rho, outer_surface_radius,
          100.0 * std::numeric_limits<double>::epsilon(),
          radius_two_outer_)) {
    return std::nullopt;
  }

  double shell_fraction_min = 0.0;
  if (target_coords[2] < z_plane_one_inner_) {
    const double cos_required = std::clamp(
        (target_coords[2] - center_one_[2]) / radius_one_, -1.0, 1.0);
    shell_fraction_min =
        (acos(cos_required) - theta_one_inner_) /
        (theta_one_outer_ - theta_one_inner_);
  }
  shell_fraction_min = std::clamp(shell_fraction_min, 0.0, 1.0);
  double shell_fraction_max = 1.0;

  const auto q = [this, &target_coords](const double shell_fraction) {
    return function_to_zero(
        shell_fraction, center_one_, center_two_, radius_one_,
        radius_two_inner_, radius_two_outer_, theta_one_inner_,
        theta_one_outer_, target_coords);
  };

  const auto make_inverse = [this, &target_coords](
                                const double shell_fraction)
      -> std::optional<std::array<double, 3>> {
    const double theta = theta_one(shell_fraction, theta_one_inner_,
                                   theta_one_outer_);
    double lambda = lambda_from_shell_fraction(
        shell_fraction, center_one_, center_two_, radius_one_,
        theta_one_inner_, theta_one_outer_, target_coords);
    constexpr double range_tolerance = 1.0e-12;
    if (lambda < -range_tolerance or lambda > 1.0 + range_tolerance) {
      return std::nullopt;
    }
    lambda = std::clamp(lambda, 0.0, 1.0);
    const double mapped_radius =
        (1.0 - lambda) * radius_one_ * sin(theta) +
        lambda * radius_two(shell_fraction, radius_two_inner_,
                            radius_two_outer_);
    const double rhobar = 1.0 + shell_fraction;
    return {{{rhobar *
                  (target_coords[0] - center_one_[0] -
                   lambda * (center_two_[0] - center_one_[0])) /
                  mapped_radius,
              rhobar *
                  (target_coords[1] - center_one_[1] -
                   lambda * (center_two_[1] - center_one_[1])) /
                  mapped_radius,
              2.0 * lambda - 1.0}}};
  };

  double q_min = q(shell_fraction_min);
  double q_max = q(shell_fraction_max);
  const double coordinate_scale =
      std::max({1.0, radius_one_, radius_two_outer_,
                abs(center_two_[2] - center_one_[2])});
  const double endpoint_tolerance =
      1.0e-13 * square(coordinate_scale);
  if (abs(q_min) <= endpoint_tolerance) {
    return make_inverse(shell_fraction_min);
  }
  if (abs(q_max) <= endpoint_tolerance) {
    return make_inverse(shell_fraction_max);
  }
  if (q_min * q_max > 0.0 or shell_fraction_min == shell_fraction_max) {
    return std::nullopt;
  }

  constexpr double abs_tol = 1.0e-15;
  constexpr double rel_tol = 1.0e-15;
  const double shell_fraction = RootFinder::toms748(
      q, shell_fraction_min, shell_fraction_max, q_min, q_max, abs_tol,
      rel_tol);
  return make_inverse(shell_fraction);
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
UniformCylindricalFlatEndcapShell::jacobian(
    const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& xbar = source_coords[0];
  const ReturnType& ybar = source_coords[1];
  const ReturnType& zbar = source_coords[2];

  const ReturnType rhobar = sqrt(square(xbar) + square(ybar));
  const ReturnType cos_phi = xbar / rhobar;
  const ReturnType sin_phi = ybar / rhobar;
  const ReturnType shell_fraction = rhobar - 1.0;
  const ReturnType lambda = 0.5 * (zbar + 1.0);
  const ReturnType theta =
      theta_one_inner_ +
      shell_fraction * (theta_one_outer_ - theta_one_inner_);
  const ReturnType flat_radius =
      radius_two_inner_ +
      shell_fraction * (radius_two_outer_ - radius_two_inner_);
  const ReturnType sphere_radius = radius_one_ * sin(theta);
  const ReturnType mapped_radius =
      (1.0 - lambda) * sphere_radius + lambda * flat_radius;
  const ReturnType d_mapped_radius_d_rhobar =
      (1.0 - lambda) * radius_one_ * cos(theta) *
          (theta_one_outer_ - theta_one_inner_) +
      lambda * (radius_two_outer_ - radius_two_inner_);
  const ReturnType angular_scale = mapped_radius / rhobar;

  auto jac = make_with_value<
      tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
      dereference_wrapper(source_coords[0]), 0.0);

  get<0, 0>(jac) = square(cos_phi) * d_mapped_radius_d_rhobar +
                   square(sin_phi) * angular_scale;
  get<0, 1>(jac) = cos_phi * sin_phi *
                   (d_mapped_radius_d_rhobar - angular_scale);
  get<1, 0>(jac) = get<0, 1>(jac);
  get<1, 1>(jac) = square(sin_phi) * d_mapped_radius_d_rhobar +
                   square(cos_phi) * angular_scale;

  const ReturnType dz_d_rhobar =
      -(1.0 - lambda) * radius_one_ * sin(theta) *
      (theta_one_outer_ - theta_one_inner_);
  get<2, 0>(jac) = cos_phi * dz_d_rhobar;
  get<2, 1>(jac) = sin_phi * dz_d_rhobar;

  get<0, 2>(jac) =
      0.5 * (center_two_[0] - center_one_[0] +
             cos_phi * (flat_radius - sphere_radius));
  get<1, 2>(jac) =
      0.5 * (center_two_[1] - center_one_[1] +
             sin_phi * (flat_radius - sphere_radius));
  get<2, 2>(jac) =
      0.5 * (center_two_[2] - center_one_[2] - radius_one_ * cos(theta));
  return jac;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
UniformCylindricalFlatEndcapShell::inv_jacobian(
    const std::array<T, 3>& source_coords) const {
  return determinant_and_inverse(jacobian(source_coords)).second;
}

void UniformCylindricalFlatEndcapShell::pup(PUP::er& p) {
  size_t version = 0;
  p | version;
  if (version >= 0) {
    p | center_one_;
    p | center_two_;
    p | radius_one_;
    p | radius_two_inner_;
    p | radius_two_outer_;
    p | z_plane_one_inner_;
    p | z_plane_one_outer_;
    p | theta_one_inner_;
    p | theta_one_outer_;
  }
}

bool operator==(const UniformCylindricalFlatEndcapShell& lhs,
                const UniformCylindricalFlatEndcapShell& rhs) {
  return lhs.center_one_ == rhs.center_one_ and
         lhs.center_two_ == rhs.center_two_ and
         lhs.radius_one_ == rhs.radius_one_ and
         lhs.radius_two_inner_ == rhs.radius_two_inner_ and
         lhs.radius_two_outer_ == rhs.radius_two_outer_ and
         lhs.z_plane_one_inner_ == rhs.z_plane_one_inner_ and
         lhs.z_plane_one_outer_ == rhs.z_plane_one_outer_;
}

bool operator!=(const UniformCylindricalFlatEndcapShell& lhs,
                const UniformCylindricalFlatEndcapShell& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>               \
  UniformCylindricalFlatEndcapShell::operator()(                             \
      const std::array<DTYPE(data), 3>& source_coords) const;                \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  UniformCylindricalFlatEndcapShell::jacobian(                               \
      const std::array<DTYPE(data), 3>& source_coords) const;                \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  UniformCylindricalFlatEndcapShell::inv_jacobian(                           \
      const std::array<DTYPE(data), 3>& source_coords) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))

#undef DTYPE
#undef INSTANTIATE

}  // namespace domain::CoordinateMaps
