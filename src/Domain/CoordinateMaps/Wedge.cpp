// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Wedge.hpp"

#include <climits>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <pup.h>

#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain::CoordinateMaps {

template <size_t Dim>
Wedge<Dim>::Wedge(const double radius_inner, const double radius_outer,
                  const double sphericity_inner, const double sphericity_outer,
                  // change to radius_bounding_cube (R_b)
                  const double cube_half_length,
                  const std::array<double, Dim> focal_offset,
                  OrientationMap<Dim> orientation_of_wedge,
                  const bool with_equiangular_map,
                  const WedgeHalves halves_to_use,
                  const Distribution radial_distribution,
                  const std::array<double, Dim - 1>& opening_angles,
                  const bool with_adapted_equiangular_map)
    : radius_inner_(radius_inner),
      radius_outer_(radius_outer),
      sphericity_inner_(sphericity_inner),
      sphericity_outer_(sphericity_outer),
      cube_half_length_(cube_half_length),
      focal_offset_(focal_offset),
      orientation_of_wedge_(std::move(orientation_of_wedge)),
      with_equiangular_map_(with_equiangular_map),
      halves_to_use_(halves_to_use),
      radial_distribution_(radial_distribution),
      opening_angles_(opening_angles) {
  const double sqrt_dim = sqrt(double{Dim});
  ASSERT(radius_inner > 0.0,
         "The radius of the inner surface must be greater than zero.");
  ASSERT(sphericity_inner >= 0.0 and sphericity_inner <= 1.0,
         "Sphericity of the inner surface must be between 0 and 1");
  ASSERT(sphericity_outer >= 0.0 and sphericity_outer <= 1.0,
         "Sphericity of the outer surface must be between 0 and 1");
  ASSERT(radius_outer > radius_inner,
         "The radius of the outer surface must be greater than the radius of "
         "the inner surface.");
  ASSERT(radius_outer *
                 ((1.0 - sphericity_outer_) / sqrt_dim + sphericity_outer_) >
             radius_inner *
                 ((1.0 - sphericity_inner) / sqrt_dim + sphericity_inner),
         "The arguments passed into the constructor for Wedge result in an "
         "object where the "
         "outer surface is pierced by the inner surface.");
  // TODO: Add assert to make sure you're only using sphericity 1 or 0 if you
  // have a focal offset
  ASSERT(radial_distribution_ == Distribution::Linear or
             (sphericity_inner_ == 1.0 and sphericity_outer_ == 1.0),
         "Only the 'Linear' radial distribution is supported for non-spherical "
         "wedges.");
  ASSERT(
      get(determinant(discrete_rotation_jacobian(orientation_of_wedge_))) > 0.0,
      "Wedge rotations must be done in such a manner that the sign of "
      "the determinant of the discrete rotation is positive. This is to "
      "preserve handedness of the coordinates.");
  ASSERT(opening_angles_ != make_array<Dim - 1>(M_PI_2) ? with_equiangular_map
                                                        : true,
         "If using opening angles other than pi/2, then the "
         "equiangular map option must be turned on.");
  if (radial_distribution_ == Distribution::Linear) {
    scaled_frustum_zero_ = 0.5 / sqrt_dim *
                           ((1.0 - sphericity_outer_) * radius_outer +
                            (1.0 - sphericity_inner) * radius_inner);
    sphere_zero_ = 0.5 * (sphericity_outer_ * radius_outer +
                          sphericity_inner * radius_inner);
    scaled_frustum_rate_ = 0.5 / sqrt_dim *
                           ((1.0 - sphericity_outer_) * radius_outer -
                            (1.0 - sphericity_inner) * radius_inner);
    sphere_rate_ = 0.5 * (sphericity_outer_ * radius_outer -
                          sphericity_inner * radius_inner);
    // TODO : change to an if else, don't want to recompute these
    if (not equal_within_roundoff(magnitude(focal_offset_), 0.0)) {
      scaled_frustum_zero_ =
          0.5 * cube_half_length_ *
          // TODO : turn into ternary expression so we don't mislead that
          // sphericity can be anything between 0 and 1 because it can't
          ((1.0 - sphericity_outer_) + (1.0 - sphericity_inner));
      scaled_frustum_rate_ =
          0.5 * cube_half_length_ *
          // TODO : turn into ternary expression so we don't mislead that
          // sphericity can be anything between 0 and 1 because it can't
          ((1.0 - sphericity_outer_) - (1.0 - sphericity_inner));
    }
  } else if (radial_distribution_ == Distribution::Logarithmic) {
    scaled_frustum_zero_ = 0.0;
    sphere_zero_ = 0.5 * (log(radius_outer * radius_inner));
    scaled_frustum_rate_ = 0.0;
    sphere_rate_ = 0.5 * (log(radius_outer / radius_inner));
  } else if (radial_distribution_ == Distribution::Inverse) {
    scaled_frustum_zero_ = 0.0;
    // Most places where sphere_zero_ and sphere_rate_ would be used
    // would cause precision issues for large wedges, so we don't
    // define them.
    // 0.5 * (radius_inner + radius_outer) / radius_inner / radius_outer;
    sphere_zero_ = std::numeric_limits<double>::signaling_NaN();
    scaled_frustum_rate_ = 0.0;
    // 0.5 * (radius_inner - radius_outer) / radius_inner / radius_outer;
    sphere_rate_ = std::numeric_limits<double>::signaling_NaN();
  } else {
    ERROR("Unsupported radial distribution: " << radial_distribution_);
  }
  if (with_adapted_equiangular_map) {
    opening_angles_distribution_ = opening_angles_;
  } else {
    opening_angles_distribution_ = make_array<Dim - 1>(M_PI_2);
  }
}

template <size_t Dim>
template <typename T>
// Change name :) to lifting_factor_lambda
tt::remove_cvref_wrap_t<T> Wedge<Dim>::default_physical_z(
    const T& zeta, const T& one_over_rho) const {
  if (radial_distribution_ == Distribution::Linear) {
    // Using auto keeps this as a blaze expression.
    const auto zeta_coefficient =
        (scaled_frustum_rate_ + sphere_rate_ * one_over_rho);
    const auto z_zero = (scaled_frustum_zero_ + sphere_zero_ * one_over_rho);
    return z_zero + zeta_coefficient * zeta;
  } else if (radial_distribution_ == Distribution::Logarithmic) {
    return exp(sphere_zero_ + sphere_rate_ * zeta) * one_over_rho;
  } else {
    return 2.0 * one_over_rho /
           ((1.0 + zeta) / radius_outer_ + (1.0 - zeta) / radius_inner_);
  }
}

template <size_t Dim>
template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, Dim> Wedge<Dim>::operator()(
    const std::array<T, Dim>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  // Radial coordinate
  const ReturnType& zeta = source_coords[radial_coord];

  // Polar angle
  ReturnType xi = source_coords[polar_coord];
  // focal_offset_[polar_coord] / cube_half_length_;
  if (halves_to_use_ == WedgeHalves::UpperOnly) {
    xi += 1.0;
    xi *= 0.5;
  } else if (halves_to_use_ == WedgeHalves::LowerOnly) {
    xi -= 1.0;
    xi *= 0.5;
  }

  auto rotated_focus =
      discrete_rotation(orientation_of_wedge_.inverse_map(), focal_offset_);

  std::array<ReturnType, Dim - 1> cap{};
  cap[0] = with_equiangular_map_
               ? tan(0.5 * opening_angles_[0]) *
                     tan(0.5 * opening_angles_distribution_[0] * xi) /
                     tan(0.5 * opening_angles_distribution_[0])
               : xi;
  ReturnType one_over_rho =
      square(1.0 - rotated_focus[radial_coord] / cube_half_length_) +
      square(cap[0] - rotated_focus[polar_coord] / cube_half_length_);
  if constexpr (Dim == 3) {
    // Azimuthal angle
    const ReturnType& eta = source_coords[azimuth_coord];
    // focal_offset_[azimuth_coord] / cube_half_length_;
    cap[1] = with_equiangular_map_
                 ? tan(0.5 * opening_angles_[1]) *
                       tan(0.5 * opening_angles_distribution_[1] * eta) /
                       tan(0.5 * opening_angles_distribution_[1])
                 : eta;
    one_over_rho +=
        square(cap[1] - rotated_focus[azimuth_coord] / cube_half_length_);
  }
  one_over_rho = 1. / sqrt(one_over_rho);

  std::array<ReturnType, Dim> physical_coords{};
  // Note this is not the full physical_coords[radial_coord] it still needs +
  // focal_offset_[2] but I specifically leave that out so I can reuse this.
  auto lifting_factor_lambda = default_physical_z(zeta, one_over_rho);
  physical_coords[radial_coord] =
      lifting_factor_lambda *
          (1.0 - rotated_focus[radial_coord] / cube_half_length_) +
      rotated_focus[radial_coord];
  physical_coords[polar_coord] =
      lifting_factor_lambda * (cap[0] - rotated_focus[0] / cube_half_length_) +
      rotated_focus[polar_coord];
  if constexpr (Dim == 3) {
    physical_coords[azimuth_coord] =
        lifting_factor_lambda *
            (cap[1] - rotated_focus[1] / cube_half_length_) +
        rotated_focus[azimuth_coord];
  }
  auto result =
      discrete_rotation(orientation_of_wedge_, std::move(physical_coords));

  return result;
}

template <size_t Dim>
std::optional<std::array<double, Dim>> Wedge<Dim>::inverse(
    const std::array<double, Dim>& target_coords) const {
  const std::array<double, Dim> physical_coords =
      discrete_rotation(orientation_of_wedge_.inverse_map(), target_coords);

  if (physical_coords[radial_coord] < 0.0 or
      equal_within_roundoff(physical_coords[radial_coord], 0.0)) {
    return std::nullopt;
  }

  const double generalized_z =
      (physical_coords[radial_coord] - focal_offset_[radial_coord]) /
      (1.0 - focal_offset_[radial_coord] / cube_half_length_);
  const double one_over_generalized_z = 1.0 / generalized_z;
  std::array<double, Dim - 1> cap{};
  cap[0] = (physical_coords[polar_coord] - focal_offset_[polar_coord]) *
               one_over_generalized_z +
           focal_offset_[polar_coord] / cube_half_length_;
  if constexpr (Dim == 3) {
    cap[1] = (physical_coords[azimuth_coord] - focal_offset_[azimuth_coord]) *
                 one_over_generalized_z +
             focal_offset_[azimuth_coord] / cube_half_length_;
  }
  const double radius = magnitude(physical_coords);
  // Radial coordinate
  double zeta = std::numeric_limits<double>::signaling_NaN();
  if (radial_distribution_ == Distribution::Linear) {
    const double one_over_rho =
        generalized_z / magnitude(physical_coords - focal_offset_);
    const double zeta_coefficient =
        (scaled_frustum_rate_ + sphere_rate_ * one_over_rho);
    // If -sphere_rate_/scaled_frustum_rate_ > 1, then
    // there exists a cone in x,y,z space given by the surface
    // zeta_coefficient=0; the map is singular on this surface.
    // We return nullopt if we are on or outside this cone.
    // If scaled_frustum_rate_ > 0, then outside the cone
    // corresponds to zeta_coefficient > 0, and if scaled_frustum_rate_
    // < 0, then outside the cone corresponds to zeta_coefficient < 0.
    // We test in two cases, and avoid division.
    if ((scaled_frustum_rate_ > 0.0 and scaled_frustum_rate_ < -sphere_rate_ and
         zeta_coefficient > 0.0) or
        (scaled_frustum_rate_ < 0.0 and scaled_frustum_rate_ > -sphere_rate_ and
         zeta_coefficient < 0.0) or
        equal_within_roundoff(zeta_coefficient, 0.0)) {
      return std::nullopt;
    }
    const auto z_zero = (scaled_frustum_zero_ + sphere_zero_ * one_over_rho);
    zeta = (generalized_z - z_zero) / zeta_coefficient;
  } else if (radial_distribution_ == Distribution::Logarithmic) {
    zeta = (log(radius) - sphere_zero_) / sphere_rate_;
  } else {
    zeta = (radius_inner_ * (radius_outer_ / radius - 1.0) +
            radius_outer_ * (radius_inner_ / radius - 1.0)) /
           (radius_inner_ - radius_outer_);
  }
  // Polar angle
  double xi = with_equiangular_map_
                  ? 2.0 *
                        atan(tan(0.5 * opening_angles_distribution_[0]) /
                             tan(0.5 * opening_angles_[0]) * cap[0]) /
                        opening_angles_distribution_[0]
                  : cap[0];
  if (halves_to_use_ == WedgeHalves::UpperOnly) {
    xi *= 2.0;
    xi -= 1.0;
  } else if (halves_to_use_ == WedgeHalves::LowerOnly) {
    xi *= 2.0;
    xi += 1.0;
  }
  std::array<double, Dim> logical_coords{};
  logical_coords[radial_coord] = zeta;
  logical_coords[polar_coord] = xi;
  if constexpr (Dim == 3) {
    logical_coords[azimuth_coord] =
        with_equiangular_map_
            ? 2.0 *
                  atan(tan(0.5 * opening_angles_distribution_[1]) /
                       tan(0.5 * opening_angles_[1]) * cap[1]) /
                  opening_angles_distribution_[1]
            : cap[1];
  }
  return logical_coords;
}

template <size_t Dim>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame> Wedge<Dim>::jacobian(
    const std::array<T, Dim>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  // std::cout << "=== In jacobian() ===" << std::endl << std::endl;
  const std::string dist_type =
      radial_distribution_ == Distribution::Linear
          ? "Distribution::Linear"
          : (radial_distribution_ == Distribution::Logarithmic
                 ? "Distribution::Logarithmic"
                 : (radial_distribution_ == Distribution::Inverse
                        ? "Distribution::Inverse"
                        : "Unknown Distribution type"));
  // std::cout << " Using Distribution type: " << dist_type << std::endl;

  // Radial coordinate
  const ReturnType& zeta = source_coords[radial_coord];

  // Polar angle
  ReturnType xi = source_coords[polar_coord];
  if (halves_to_use_ == WedgeHalves::UpperOnly) {
    xi += 1.0;
    xi *= 0.5;
  } else if (halves_to_use_ == WedgeHalves::LowerOnly) {
    xi -= 1.0;
    xi *= 0.5;
  }
  std::array<ReturnType, Dim - 1> cap{};
  std::array<ReturnType, Dim - 1> cap_deriv{};

  cap[0] = with_equiangular_map_
               ? tan(0.5 * opening_angles_[0]) *
                     tan(0.5 * opening_angles_distribution_[0] * xi) /
                     tan(0.5 * opening_angles_distribution_[0])
               : xi;
  cap_deriv[0] =
      with_equiangular_map_
          ? 0.5 * opening_angles_distribution_[0] *
                tan(0.5 * opening_angles_[0]) /
                tan(0.5 * opening_angles_distribution_[0]) /
                square(cos(0.5 * opening_angles_distribution_[0] * xi))
          : make_with_value<ReturnType>(xi, 1.0);

  const auto rotated_focus =
      discrete_rotation(orientation_of_wedge_.inverse_map(), focal_offset_);

  // std::cout << "focal_offset_[polar_coord] " << focal_offset_[polar_coord]
  //           << std::endl;
  // std::cout << "rotated_focus[polar_coord] " << rotated_focus[polar_coord]
  //           << std::endl;
  // if constexpr (Dim == 3) {
  //   std::cout << "rotated_focus[azimuth_coord] " <<
  //   rotated_focus[azimuth_coord]
  //             << std::endl;
  //   std::cout << "focal_offset_[azimuth_coord] " <<
  //   focal_offset_[azimuth_coord]
  //             << std::endl;
  // }
  // std::cout << "focal_offset_[radial_coord] " << focal_offset_[radial_coord]
  //           << std::endl;
  // std::cout << "rotated_focus[radial_coord] " << rotated_focus[radial_coord]
  //           << std::endl;

  // const std::array<double, 3> gamma {{
  //   cap[0] - rotated_focus[polar_coord] / cube_half_length_,
  //   cap[1] - rotated_focus[polar_coord] / cube_half_length_,
  //   1.0 - rotated_focus[polar_coord] / cube_half_length_
  // }};
  std::array<ReturnType, Dim> gamma{};
  gamma[polar_coord] = cap[0] - rotated_focus[polar_coord] / cube_half_length_;
  gamma[radial_coord] = make_with_value<ReturnType>(cap[0], 1.0) -
                        rotated_focus[radial_coord] / cube_half_length_;
  // std::cout << "gamma[polar_coord] " << gamma[polar_coord] << std::endl;
  // std::cout << "gamma[radial_coord] " << gamma[radial_coord] << std::endl;

  // ReturnType one_over_rho = 1.0 + square(cap[0]);
  ReturnType one_over_rho =
      square(1.0 - rotated_focus[radial_coord] / cube_half_length_) +
      square(cap[0] - rotated_focus[polar_coord] / cube_half_length_);

  if constexpr (Dim == 3) {
    // Azimuthal angle
    const ReturnType& eta = source_coords[azimuth_coord];
    cap[1] = with_equiangular_map_
                 ? tan(0.5 * opening_angles_[1]) *
                       tan(0.5 * opening_angles_distribution_[1] * eta) /
                       tan(0.5 * opening_angles_distribution_[1])
                 : eta;
    cap_deriv[1] =
        with_equiangular_map_
            ? 0.5 * opening_angles_distribution_[1] *
                  tan(0.5 * opening_angles_[1]) /
                  tan(0.5 * opening_angles_distribution_[1]) /
                  square(cos(0.5 * opening_angles_distribution_[1] * eta))
            : make_with_value<ReturnType>(xi, 1.0);

    gamma[azimuth_coord] =
        cap[1] - rotated_focus[azimuth_coord] / cube_half_length_;

    // std::cout << "gamma[azimuth_coord] " << gamma[azimuth_coord] <<
    // std::endl;

    // one_over_rho += square(cap[1]);
    one_over_rho +=
        square(cap[1] - rotated_focus[azimuth_coord] / cube_half_length_);
  }
  one_over_rho = 1. / sqrt(one_over_rho);

  // std::cout << "one_over_rho: " << one_over_rho << std::endl;

  const ReturnType s_factor = [this, &zeta]() -> ReturnType {
    if (radial_distribution_ == Distribution::Linear) {
      return sphere_zero_ + sphere_rate_ * zeta;
    } else if (radial_distribution_ == Distribution::Logarithmic) {
      return exp(sphere_zero_ + sphere_rate_ * zeta);
    } else {
      return 2.0 /
             ((1.0 + zeta) / radius_outer_ + (1.0 - zeta) / radius_inner_);
    }
  }();

  const ReturnType s_factor_deriv = [this, &zeta, &s_factor]() -> ReturnType {
    if (radial_distribution_ == Distribution::Linear) {
      // return sphere_rate_;
      return make_with_value<ReturnType>(zeta, sphere_rate_);
    } else if (radial_distribution_ == Distribution::Logarithmic) {
      // return 0.5 *
      //        ((1.0 + zeta) * pow(radius_outer_, (1.0 - zeta)) +
      //         (1.0 - zeta) * pow(radius_inner_, (1.0 + zeta))) /
      //        s_factor;
      // // TODO : do something better than this:
      // ReturnType radius_outer_factor = make_with_value<ReturnType>(zeta,
      // 0.0); ReturnType radius_inner_factor =
      // make_with_value<ReturnType>(zeta, 0.0); if constexpr
      // (is_derived_of_vector_impl_v<ReturnType>) {
      //   for (size_t i = 0; i < zeta.size(); i++) {
      //     radius_outer_factor[i] = pow(radius_outer_, (1.0 - zeta[i]));
      //     radius_inner_factor[i] = pow(radius_inner_, (1.0 + zeta[i]));
      //   }
      // } else {
      //   radius_outer_factor = pow(radius_outer_, (1.0 - zeta));
      //   radius_inner_factor = pow(radius_inner_, (1.0 + zeta));
      // }
      // return 0.5 *
      //        ((1.0 + zeta) * radius_outer_factor +
      //         (1.0 - zeta) * radius_inner_factor) /
      //        s_factor;
      return 0.5 * s_factor * log(radius_outer_ / radius_inner_);
    } else {
      return 2.0 *
             ((square(radius_outer_) * radius_inner_) -
              radius_outer_ * square(radius_inner_)) /
             square(radius_outer_ + radius_inner_ +
                    zeta * (radius_inner_ - radius_outer_));
    }
  }();

  // std::cout << "s_factor: " << s_factor << std::endl;

  const ReturnType one_over_rho_cubed = pow<3>(one_over_rho);
  const ReturnType s_factor_over_rho_cubed = s_factor * one_over_rho_cubed;
  const ReturnType lifting_factor_lambda =
      default_physical_z(zeta, one_over_rho);

  // std::cout << "sphere_rate_: " << sphere_rate_ << std::endl;
  // std::cout << "scaled_frustum_rate_: " << scaled_frustum_rate_ << std::endl;

  // auto d_lifting_factor_lambda = make_with_value<tnsr::i<ReturnType, Dim,
  // Frame::NoFrame>>(
  //     xi,
  //     std::numeric_limits<double>::signaling_NaN());
  std::array<ReturnType, Dim> d_lifting_factor_lambda{};
  d_lifting_factor_lambda[polar_coord] =
      -s_factor_over_rho_cubed * cap_deriv[0] * gamma[polar_coord];
  if (radial_distribution_ == Distribution::Linear) {
    // TODO : make S' function that returns something different for each
    // distribution, where for linear it's just sphere_rate_, or make a
    // function for computing this d_lifting_factor_lambda
    d_lifting_factor_lambda[radial_coord] =
        sphere_rate_ * one_over_rho + scaled_frustum_rate_;
  } else if (radial_distribution_ == Distribution::Logarithmic) {
    // TODO : need to implement this case? not equal to Linear case above?
    //
    // d_lifting_factor_lambda[radial_coord] = S' * one_over_rho;
    d_lifting_factor_lambda[radial_coord] = s_factor_deriv * one_over_rho;
  } else {
    // d_lifting_factor_lambda[radial_coord] =
    // TODO : can't just use sphere_rate_ = (0.5 * (radius_inner - radius_outer)
    // / radius_inner / radius_outer) sphere_rate_ * one_over_rho; (0.5 *
    // (radius_inner - radius_outer) / radius_inner / radius_outer) *
    // one_over_rho;
    //
    // d_lifting_factor_lambda[radial_coord] = S' * one_over_rho;
    d_lifting_factor_lambda[radial_coord] = s_factor_deriv * one_over_rho;
  }
  // std::cout << "d_lifting_factor_lambda[polar_coord] "
  //           << d_lifting_factor_lambda[polar_coord] << std::endl;
  // std::cout << "d_lifting_factor_lambda[radial_coord] "
  //           << d_lifting_factor_lambda[radial_coord] << std::endl;
  if (Dim == 3) {
    d_lifting_factor_lambda[azimuth_coord] =
        -s_factor_over_rho_cubed * cap_deriv[1] * gamma[azimuth_coord];
    // std::cout << "d_lifting_factor_lambda[azimuth_coord] "
    //           << d_lifting_factor_lambda[azimuth_coord] << std::endl;
  }

  auto jacobian_matrix =
      make_with_value<tnsr::Ij<ReturnType, Dim, Frame::NoFrame>>(xi, 0.0);

  // Derivative by polar angle
  std::array<ReturnType, Dim> dxyz_dxi{};
  dxyz_dxi[radial_coord] =
      gamma[radial_coord] * d_lifting_factor_lambda[polar_coord];
  // std::cout << "dxyz_dxi[radial_coord]: " << dxyz_dxi[radial_coord]
  //           << std::endl;
  if (radial_distribution_ == Distribution::Linear) {
    dxyz_dxi[polar_coord] =
        gamma[polar_coord] * d_lifting_factor_lambda[polar_coord] +
        cap_deriv[0] * lifting_factor_lambda;
    // std::cout << "dxyz_dxi[polar_coord]: " << dxyz_dxi[polar_coord]
    //           << std::endl;
  } else {
    dxyz_dxi[polar_coord] =
        square(one_over_rho) * cap_deriv[0] * lifting_factor_lambda;
    // std::cout << "dxyz_dxi[polar_coord]: " << dxyz_dxi[polar_coord]
    //           << std::endl;
    if constexpr (Dim == 3) {
      dxyz_dxi[polar_coord] *= 1.0 + square(cap[1]);
      // std::cout << "dxyz_dxi[polar_coord]: " << dxyz_dxi[polar_coord]
      //           << std::endl;
    }
  }
  if constexpr (Dim == 3) {
    dxyz_dxi[azimuth_coord] =
        gamma[azimuth_coord] * d_lifting_factor_lambda[polar_coord];
    // std::cout << "dxyz_dxi[azimuth_coord]: " << dxyz_dxi[azimuth_coord]
    //           << std::endl;
  }
  // Implement Scalings:
  if (halves_to_use_ != WedgeHalves::Both) {
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(dxyz_dxi, d) *= 0.5;
    }
  }
  // std::cout << "dxyz_dxi[polar_coord]: " << dxyz_dxi[polar_coord] <<
  // std::endl;
  std::array<ReturnType, Dim> dX_dlogical =
      discrete_rotation(orientation_of_wedge_, std::move(dxyz_dxi));
  get<0, polar_coord>(jacobian_matrix) = dX_dlogical[0];
  get<1, polar_coord>(jacobian_matrix) = dX_dlogical[1];
  if constexpr (Dim == 3) {
    get<2, polar_coord>(jacobian_matrix) = dX_dlogical[2];
  }

  // std::cout << "=== completed rotating and setting dx_dlogical ==="
  //           << std::endl;

  // Derivative by azimuthal angle
  if constexpr (Dim == 3) {
    std::array<ReturnType, Dim> dxyz_deta{};
    dxyz_deta[radial_coord] =
        gamma[radial_coord] * d_lifting_factor_lambda[azimuth_coord];
    // std::cout << "dxyz_deta[radial_coord]: " << dxyz_deta[radial_coord]
    //           << std::endl;

    if (radial_distribution_ == Distribution::Linear) {
      dxyz_deta[azimuth_coord] =
          gamma[azimuth_coord] * d_lifting_factor_lambda[azimuth_coord] +
          cap_deriv[1] * lifting_factor_lambda;
      // std::cout << "dxyz_deta[azimuth_coord]: " << dxyz_deta[azimuth_coord]
      //           << std::endl;
    } else {
      dxyz_deta[azimuth_coord] = (1.0 + square(cap[0])) * square(one_over_rho) *
                                 cap_deriv[1] * lifting_factor_lambda;
      // std::cout << "dxyz_deta[azimuth_coord]: " << dxyz_deta[azimuth_coord]
      //           << std::endl;
    }

    dxyz_deta[polar_coord] =
        gamma[polar_coord] * d_lifting_factor_lambda[azimuth_coord];
    // std::cout << "dxyz_deta[polar_coord]: " << dxyz_deta[polar_coord]
    //           << std::endl;
    dX_dlogical =
        discrete_rotation(orientation_of_wedge_, std::move(dxyz_deta));
    get<0, azimuth_coord>(jacobian_matrix) = dX_dlogical[0];
    get<1, azimuth_coord>(jacobian_matrix) = dX_dlogical[1];
    get<2, azimuth_coord>(jacobian_matrix) = dX_dlogical[2];

    // std::cout << "=== completed rotating and setting dy_dlogical ==="
    //           << std::endl;
  }

  // Derivative by radial coordinate
  std::array<ReturnType, Dim> dxyz_dzeta{};
  if (radial_distribution_ == Distribution::Linear) {
    dxyz_dzeta[radial_coord] =
        gamma[radial_coord] * d_lifting_factor_lambda[radial_coord];
    // std::cout << "dxyz_dzeta[radial_coord]: " << dxyz_dzeta[radial_coord]
    //           << std::endl;
  } else if (radial_distribution_ == Distribution::Logarithmic) {
    dxyz_dzeta[radial_coord] = s_factor * sphere_rate_ * one_over_rho;
    // std::cout << "dxyz_dzeta[radial_coord]: " << dxyz_dzeta[radial_coord]
    //           << std::endl;
  } else {
    const double sphere_rate =
        0.5 * (1.0 / radius_outer_ - 1.0 / radius_inner_);
    dxyz_dzeta[radial_coord] = -square(s_factor) * sphere_rate * one_over_rho;
    // std::cout << "dxyz_dzeta[radial_coord]: " << dxyz_dzeta[radial_coord]
    //           << std::endl;
  }
  dxyz_dzeta[polar_coord] =
      gamma[polar_coord] * d_lifting_factor_lambda[radial_coord];
  // std::cout << "dxyz_dzeta[polar_coord]: " << dxyz_dzeta[polar_coord]
  //           << std::endl;

  if constexpr (Dim == 3) {
    dxyz_dzeta[azimuth_coord] =
        gamma[azimuth_coord] * d_lifting_factor_lambda[radial_coord];
    // std::cout << "dxyz_dzeta[azimuth_coord]: " << dxyz_dzeta[azimuth_coord]
    //           << std::endl;
  }

  // std::cout << "dxyz_dzeta[polar_coord]: " << dxyz_dzeta[polar_coord]
  //           << std::endl;
  // std::cout << "dxyz_dzeta[radial_coord]: " << dxyz_dzeta[radial_coord]
  //           << std::endl;

  dX_dlogical = discrete_rotation(orientation_of_wedge_, std::move(dxyz_dzeta));
  get<0, radial_coord>(jacobian_matrix) = dX_dlogical[0];
  get<1, radial_coord>(jacobian_matrix) = dX_dlogical[1];
  if constexpr (Dim == 3) {
    get<2, radial_coord>(jacobian_matrix) = dX_dlogical[2];
  }
  // std::cout << "=== completed rotating and setting dz_dlogical ==="
  //           << std::endl;
  return jacobian_matrix;
}

template <size_t Dim>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame>
Wedge<Dim>::inv_jacobian(const std::array<T, Dim>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  // Radial coordinate
  const ReturnType& zeta = source_coords[radial_coord];

  // Polar angle
  ReturnType xi = source_coords[polar_coord];
  if (halves_to_use_ == WedgeHalves::UpperOnly) {
    xi += 1.0;
    xi *= 0.5;
  } else if (halves_to_use_ == WedgeHalves::LowerOnly) {
    xi -= 1.0;
    xi *= 0.5;
  }
  std::array<ReturnType, Dim> cap{};
  std::array<ReturnType, Dim> cap_deriv{};
  cap[0] = with_equiangular_map_
               ? tan(0.5 * opening_angles_[0]) *
                     tan(0.5 * opening_angles_distribution_[0] * xi) /
                     tan(0.5 * opening_angles_distribution_[0])
               : xi;
  cap_deriv[0] =
      with_equiangular_map_
          ? 0.5 * opening_angles_distribution_[0] *
                tan(0.5 * opening_angles_[0]) /
                tan(0.5 * opening_angles_distribution_[0]) /
                square(cos(0.5 * opening_angles_distribution_[0] * xi))
          : make_with_value<ReturnType>(xi, 1.0);

  const auto rotated_focus =
      discrete_rotation(orientation_of_wedge_.inverse_map(), focal_offset_);

  std::array<ReturnType, Dim> gamma{};
  gamma[polar_coord] = cap[0] - rotated_focus[polar_coord] / cube_half_length_;
  gamma[radial_coord] = make_with_value<ReturnType>(cap[0], 1.0) -
                        rotated_focus[radial_coord] / cube_half_length_;

  // ReturnType one_over_rho = 1.0 + square(cap[0]);
  ReturnType one_over_rho =
      square(1.0 - rotated_focus[radial_coord] / cube_half_length_) +
      square(cap[0] - rotated_focus[polar_coord] / cube_half_length_);
  if constexpr (Dim == 3) {
    // Azimuthal angle
    const ReturnType& eta = source_coords[azimuth_coord];
    cap[1] = with_equiangular_map_
                 ? tan(0.5 * opening_angles_[1]) *
                       tan(0.5 * opening_angles_distribution_[1] * eta) /
                       tan(0.5 * opening_angles_distribution_[1])
                 : eta;
    cap_deriv[1] =
        with_equiangular_map_
            ? 0.5 * opening_angles_distribution_[1] *
                  tan(0.5 * opening_angles_[1]) /
                  tan(0.5 * opening_angles_distribution_[1]) /
                  square(cos(0.5 * opening_angles_distribution_[1] * eta))
            : make_with_value<ReturnType>(xi, 1.0);

    gamma[azimuth_coord] =
        cap[1] - rotated_focus[azimuth_coord] / cube_half_length_;

    // one_over_rho += square(cap[1]);
    one_over_rho +=
        square(cap[1] - rotated_focus[azimuth_coord] / cube_half_length_);
  }
  one_over_rho = 1. / sqrt(one_over_rho);

  // std::cout << "one_over_rho : " << one_over_rho << std::endl;

  const ReturnType one_over_rho_cubed = pow<3>(one_over_rho);
  // const ReturnType scaled_z_frustum =
  //     scaled_frustum_zero_ + scaled_frustum_rate_ * zeta;
  const ReturnType s_factor = [this, &zeta]() -> ReturnType {
    if (radial_distribution_ == Distribution::Linear) {
      return (sphere_zero_ + sphere_rate_ * zeta);
    } else if (radial_distribution_ == Distribution::Logarithmic) {
      return exp(sphere_zero_ + sphere_rate_ * zeta);
    } else {
      return 2.0 /
             ((1.0 + zeta) / radius_outer_ + (1.0 - zeta) / radius_inner_);
    }
  }();
  const ReturnType s_factor_over_rho_cubed = s_factor * one_over_rho_cubed;
  // const ReturnType one_over_dz_dzeta = [this, &zeta,
  //                                       &one_over_rho]() -> ReturnType {
  //   if (radial_distribution_ == Distribution::Linear) {
  //     return 1.0 / (scaled_frustum_rate_ + sphere_rate_ * one_over_rho);
  //   } else if (radial_distribution_ == Distribution::Logarithmic) {
  //     return 1.0 / (exp(sphere_zero_ + sphere_rate_ * zeta) * sphere_rate_ *
  //                   one_over_rho);
  //   } else {
  //     const double sphere_rate =
  //         0.5 * (1.0 / radius_outer_ - 1.0 / radius_inner_);
  //     return -0.25 *
  //            square((1.0 + zeta) / radius_outer_ +
  //                   (1.0 - zeta) / radius_inner_) /
  //            sphere_rate / one_over_rho;
  //   }
  // }();
  // const ReturnType dzeta_factor = one_over_physical_z * one_over_dz_dzeta;

  // 1 / c
  // const ReturnType one_over_d_lifting_factor_dzeta = [this, &zeta,
  //                                       &one_over_rho]() -> ReturnType {
  //   if (radial_distribution_ == Distribution::Linear) {
  //     return 1.0 / (scaled_frustum_rate_ + sphere_rate_ * one_over_rho);
  //   } else if (radial_distribution_ == Distribution::Logarithmic) {
  //     return 1.0 / (exp(sphere_zero_ + sphere_rate_ * zeta) * sphere_rate_ *
  //                   one_over_rho);
  //   } else {
  //     const double sphere_rate =
  //         0.5 * (1.0 / radius_outer_ - 1.0 / radius_inner_);
  //     return -0.25 *
  //            square((1.0 + zeta) / radius_outer_ +
  //                   (1.0 - zeta) / radius_inner_) /
  //            sphere_rate / one_over_rho;
  //   }
  // }();

  // 1 / z
  const ReturnType one_over_gamma_z = 1.0 / gamma[radial_coord];

  const ReturnType lifting_factor_lambda =
      default_physical_z(zeta, one_over_rho);

  // // TODO : do something more optimized than this, but doing this for now
  // const ReturnType s_factor = s_factor_over_rho_cubed *

  // std::cout << "lifting_factor_lambda : " << lifting_factor_lambda <<
  // std::endl;
  const ReturnType s_factor_deriv = [this, &zeta, &s_factor]() -> ReturnType {
    if (radial_distribution_ == Distribution::Linear) {
      // return sphere_rate_;
      return make_with_value<ReturnType>(zeta, sphere_rate_);
    } else if (radial_distribution_ == Distribution::Logarithmic) {
      // return 0.5 *
      //        ((1.0 + zeta) * pow(radius_outer_, (1.0 - zeta)) +
      //         (1.0 - zeta) * pow(radius_inner_, (1.0 + zeta))) /
      //        s_factor;
      // // TODO : do something better than this:
      // ReturnType radius_outer_factor = make_with_value<ReturnType>(zeta,
      // 0.0); ReturnType radius_inner_factor =
      // make_with_value<ReturnType>(zeta, 0.0); if constexpr
      // (is_derived_of_vector_impl_v<ReturnType>) {
      //   for (size_t i = 0; i < zeta.size(); i++) {
      //     radius_outer_factor[i] = pow(radius_outer_, (1.0 - zeta[i]));
      //     radius_inner_factor[i] = pow(radius_inner_, (1.0 + zeta[i]));
      //   }
      // } else {
      //   radius_outer_factor = pow(radius_outer_, (1.0 - zeta));
      //   radius_inner_factor = pow(radius_inner_, (1.0 + zeta));
      // }
      // return 0.5 *
      //        ((1.0 + zeta) * radius_outer_factor +
      //         (1.0 - zeta) * radius_inner_factor) /
      //        s_factor;
      return 0.5 * s_factor * log(radius_outer_ / radius_inner_);
    } else {
      return 2.0 *
             ((square(radius_outer_) * radius_inner_) -
              radius_outer_ * square(radius_inner_)) /
             square(radius_outer_ + radius_inner_ +
                    zeta * (radius_inner_ - radius_outer_));
    }
  }();

  // a, b, c
  std::array<ReturnType, Dim> d_lifting_factor_lambda{};
  d_lifting_factor_lambda[polar_coord] =
      -s_factor_over_rho_cubed * cap_deriv[0] * gamma[polar_coord];

  // d_lifting_factor_lambda[radial_coord] =
  //     sphere_rate_ * one_over_rho + scaled_frustum_rate_;

  if (radial_distribution_ == Distribution::Linear) {
    d_lifting_factor_lambda[radial_coord] =
        sphere_rate_ * one_over_rho + scaled_frustum_rate_;
  } else if (radial_distribution_ == Distribution::Logarithmic) {
    d_lifting_factor_lambda[radial_coord] = s_factor_deriv * one_over_rho;
  } else {
    d_lifting_factor_lambda[radial_coord] = s_factor_deriv * one_over_rho;
  }

  if constexpr (Dim == 3) {
    d_lifting_factor_lambda[azimuth_coord] =
        -s_factor_over_rho_cubed * cap_deriv[1] * gamma[azimuth_coord];
  }

  const ReturnType one_over_d_lifting_factor_lambda_dzeta =
      1.0 / d_lifting_factor_lambda[radial_coord];

  // std::cout << "one_over_d_lifting_factor_lambda_dzeta : "
  //           << one_over_d_lifting_factor_lambda_dzeta << std::endl;

  auto inv_jacobian_matrix =
      make_with_value<tnsr::Ij<ReturnType, Dim, Frame::NoFrame>>(xi, 0.0);

  // Derivatives of polar angle
  std::array<ReturnType, Dim> dxi_dxyz{};
  // dxi_dxyz[polar_coord] = one_over_lifting_factor_lambda / cap_deriv[0];
  // std::cout << "lifting_factor_lambda : " << lifting_factor_lambda << std::endl;
  // std::cout << "cap_deriv[0] : " << cap_deriv[0] << std::endl;
  dxi_dxyz[polar_coord] = 1.0 / (lifting_factor_lambda * cap_deriv[0]);
  // std::cout << "dxi_dxyz[polar_coord] : " << dxi_dxyz[polar_coord] << std::endl;
  // // Implement Scalings:
  if (halves_to_use_ != WedgeHalves::Both) {
    dxi_dxyz[polar_coord] *= 2.0;
  }
  // dxi_dxyz[radial_coord] = -cap[0] * dxi_dxyz[polar_coord];
  // std::cout << "dxi_dxyz[polar_coord]: " << dxi_dxyz[polar_coord] <<
  // std::endl; std::cout << "one_over_d_lifting_factor_lambda_dzeta: " <<
  // one_over_d_lifting_factor_lambda_dzeta << std::endl; std::cout <<
  // "d_lifting_factor_lambda[polar_coord]: " <<
  // d_lifting_factor_lambda[polar_coord] << std::endl;
  // backwards:
  // dxi_dxyz[radial_coord] = -dxi_dxyz[polar_coord] *
  //                          one_over_d_lifting_factor_lambda_dzeta *
  //                          d_lifting_factor_lambda[polar_coord];
  dxi_dxyz[radial_coord] =
      -dxi_dxyz[polar_coord] * one_over_gamma_z * gamma[polar_coord];
  // std::cout << "dxi_dxyz[radial_coord] : " << dxi_dxyz[radial_coord] << std::endl;
  // std::cout << "dxi_dxyz[radial_coord]: " << dxi_dxyz[polar_coord] <<
  // std::endl;
  if constexpr (Dim == 3) {
    dxi_dxyz[azimuth_coord] = make_with_value<ReturnType>(xi, 0.0);
    // std::cout << "dxi_dxyz[azimuth_coord] : " << dxi_dxyz[azimuth_coord] << std::endl;
  }

  // std::cout << "dxi_dxyz[polar_coord] : " << dxi_dxyz[polar_coord] <<
  // std::endl; std::cout << "dxi_dxyz[radial_coord] : " <<
  // dxi_dxyz[radial_coord]
  //           << std::endl;

  // std::array<ReturnType, Dim> dlogical_dX =
  //     discrete_rotation(orientation_of_wedge_, std::move(dxi_dxyz));
  // get<polar_coord, 0>(inv_jacobian_matrix) = dlogical_dX[0];
  // get<polar_coord, 1>(inv_jacobian_matrix) = dlogical_dX[1];
  // if constexpr (Dim == 3) {
  //   get<polar_coord, 2>(inv_jacobian_matrix) = dlogical_dX[2];
  // }

  // std::cout << "dlogical_dX done" << std::endl;

  // Derivatives of azimuthal angle
  std::array<ReturnType, Dim> deta_dxyz{};
  (void)deta_dxyz;
  if constexpr (Dim == 3) {
    // std::array<ReturnType, Dim> deta_dxyz{};
    deta_dxyz[polar_coord] = make_with_value<ReturnType>(xi, 0.0);
    // deta_dxyz[azimuth_coord] = one_over_lifting_factor_lambda / cap_deriv[1];
    deta_dxyz[azimuth_coord] = 1.0 / (lifting_factor_lambda * cap_deriv[1]);
    // deta_dxyz[radial_coord] = -cap[1] * deta_dxyz[azimuth_coord];
    // backwards :
    // deta_dxyz[radial_coord] = -deta_dxyz[azimuth_coord] *
    //                           one_over_d_lifting_factor_lambda_dzeta *
    //                           d_lifting_factor_lambda[azimuth_coord];
    deta_dxyz[radial_coord] =
      -deta_dxyz[azimuth_coord] * one_over_gamma_z * gamma[azimuth_coord];
    // dlogical_dX =
    //     discrete_rotation(orientation_of_wedge_, std::move(deta_dxyz));
    // get<azimuth_coord, 0>(inv_jacobian_matrix) = dlogical_dX[0];
    // get<azimuth_coord, 1>(inv_jacobian_matrix) = dlogical_dX[1];
    // get<azimuth_coord, 2>(inv_jacobian_matrix) = dlogical_dX[2];

    // std::cout << "deta_dxyz[polar_coord] : " << deta_dxyz[polar_coord] << std::endl;
    // std::cout << "deta_dxyz[azimuth_coord] : " << deta_dxyz[azimuth_coord] << std::endl;
    // std::cout << "deta_dxyz[radial_coord] : " << deta_dxyz[radial_coord] << std::endl;

    // std::cout << "dlogical_dY done" << std::endl;
  }

  // Derivatives of radial coordinate
  std::array<ReturnType, Dim> dzeta_dxyz{};
  // dzeta_dxyz[radial_coord] =
  //     dzeta_factor * (scaled_z_frustum + s_factor_over_rho_cubed);
  // dzeta_dxyz[polar_coord] = dzeta_factor * cap[0] * s_factor_over_rho_cubed;

  // backwards:
  // dzeta_dxyz[polar_coord] =
  //     -dxi_dxyz[polar_coord] * one_over_gamma_z * gamma[radial_coord];
  dzeta_dxyz[polar_coord] =
      -dxi_dxyz[polar_coord] * one_over_d_lifting_factor_lambda_dzeta * d_lifting_factor_lambda[polar_coord];
  if (halves_to_use_ != WedgeHalves::Both) {
    dzeta_dxyz[polar_coord] *= 0.5;
  }
  // std::cout << "dzeta_dxyz[polar_coord] : " << dzeta_dxyz[polar_coord]
  //           << std::endl;

  if constexpr (Dim == 2) {
    // std::cout << "dzeta_dxyz[radial_coord] before" << std::endl;
    // std::cout << "one_over_gamma_z: " << one_over_gamma_z << std::endl;
    // std::cout << "one_over_d_lifting_factor_lambda_dzeta: " <<
    // one_over_d_lifting_factor_lambda_dzeta << std::endl; std::cout <<
    // "dxi_dxyz[radial_coord]: " << dxi_dxyz[radial_coord] << std::endl;
    dzeta_dxyz[radial_coord] =
        one_over_gamma_z *
        (one_over_d_lifting_factor_lambda_dzeta - dzeta_dxyz[polar_coord] * gamma[polar_coord]);
    // std::cout << "dzeta_dxyz[radial_coord] : " << dzeta_dxyz[radial_coord] <<
    // std::endl; dzeta_dxyz[radial_coord] =
    //     one_over_gamma_z *
    //     (one_over_d_lifting_factor_lambda_dzeta - get<radial_coord,
    //     0>(inv_jacobian_matrix));
    // std::cout << "dzeta_dxyz[radial_coord] after" << std::endl;
    // std::cout << "dzeta_dxyz[radial_coord] : " << dzeta_dxyz[radial_coord]
    //           << std::endl;
  } else {
    // backwards:
    // dzeta_dxyz[azimuth_coord] =
    //     -deta_dxyz[azimuth_coord] * one_over_gamma_z * gamma[azimuth_coord];
    dzeta_dxyz[azimuth_coord] =
        -deta_dxyz[azimuth_coord] * one_over_d_lifting_factor_lambda_dzeta * d_lifting_factor_lambda[azimuth_coord];
    // std::cout << "dzeta_dxyz[azimuth_coord] : " << dzeta_dxyz[azimuth_coord] << std::endl;
    dzeta_dxyz[radial_coord] =
        one_over_gamma_z * (one_over_d_lifting_factor_lambda_dzeta -
                            dzeta_dxyz[azimuth_coord] * gamma[azimuth_coord] - dzeta_dxyz[polar_coord] * gamma[polar_coord]);
    // std::cout << "dzeta_dxyz[radial_coord] : " << dzeta_dxyz[radial_coord]
    //           << std::endl;
  }

  // if constexpr (Dim == 3) {
  //   dzeta_dxyz[azimuth_coord] = dzeta_factor * cap[1] *
  //   s_factor_over_rho_cubed;
  // }

  std::array<ReturnType, Dim> dlogical_dX =
      discrete_rotation(orientation_of_wedge_, std::move(dxi_dxyz));
  get<polar_coord, 0>(inv_jacobian_matrix) = dlogical_dX[0];
  get<polar_coord, 1>(inv_jacobian_matrix) = dlogical_dX[1];
  if constexpr (Dim == 3) {
    get<polar_coord, 2>(inv_jacobian_matrix) = dlogical_dX[2];
  }

  if constexpr (Dim == 3) {
    dlogical_dX =
        discrete_rotation(orientation_of_wedge_, std::move(deta_dxyz));
    get<azimuth_coord, 0>(inv_jacobian_matrix) = dlogical_dX[0];
    get<azimuth_coord, 1>(inv_jacobian_matrix) = dlogical_dX[1];
    get<azimuth_coord, 2>(inv_jacobian_matrix) = dlogical_dX[2];
    // std::cout << "dlogical_dY done" << std::endl;
  }

  dlogical_dX = discrete_rotation(orientation_of_wedge_, std::move(dzeta_dxyz));
  get<radial_coord, 0>(inv_jacobian_matrix) = dlogical_dX[0];
  get<radial_coord, 1>(inv_jacobian_matrix) = dlogical_dX[1];
  if constexpr (Dim == 3) {
    get<radial_coord, 2>(inv_jacobian_matrix) = dlogical_dX[2];
  }
  // std::cout << "dlogical_dZ done" << std::endl;
  return inv_jacobian_matrix;
}

template <size_t Dim>
void Wedge<Dim>::pup(PUP::er& p) {
  size_t version = 2;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | radius_inner_;
    p | radius_outer_;
    p | sphericity_inner_;
    p | sphericity_outer_;
    p | orientation_of_wedge_;
    p | with_equiangular_map_;
    p | halves_to_use_;
    p | radial_distribution_;
    p | scaled_frustum_zero_;
    p | sphere_zero_;
    p | scaled_frustum_rate_;
    p | sphere_rate_;
    if (version == 0) {
      double half_opening_angle = std::numeric_limits<double>::signaling_NaN();
      p | half_opening_angle;
      opening_angles_[0] = 2. * half_opening_angle;
      if constexpr (Dim == 3) {
        opening_angles_[1] = M_PI_2;
      }
      opening_angles_distribution_ = opening_angles_;
    }
  }
  if (version >= 1) {
    p | opening_angles_;
    p | opening_angles_distribution_;
  }
  if (version >= 2) {
    p | cube_half_length_;
    p | focal_offset_;
  }
}

template <size_t Dim>
bool operator==(const Wedge<Dim>& lhs, const Wedge<Dim>& rhs) {
  return lhs.radius_inner_ == rhs.radius_inner_ and
         lhs.radius_outer_ == rhs.radius_outer_ and
         lhs.cube_half_length_ == rhs.cube_half_length_ and
         lhs.focal_offset_ == rhs.focal_offset_ and
         lhs.orientation_of_wedge_ == rhs.orientation_of_wedge_ and
         lhs.with_equiangular_map_ == rhs.with_equiangular_map_ and
         lhs.halves_to_use_ == rhs.halves_to_use_ and
         lhs.radial_distribution_ == rhs.radial_distribution_ and
         lhs.sphericity_inner_ == rhs.sphericity_inner_ and
         lhs.sphericity_outer_ == rhs.sphericity_outer_ and
         lhs.scaled_frustum_zero_ == rhs.scaled_frustum_zero_ and
         lhs.scaled_frustum_rate_ == rhs.scaled_frustum_rate_ and
         lhs.opening_angles_ == rhs.opening_angles_ and
         lhs.opening_angles_distribution_ == rhs.opening_angles_distribution_;
}

template <size_t Dim>
bool operator!=(const Wedge<Dim>& lhs, const Wedge<Dim>& rhs) {
  return not(lhs == rhs);
}

// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_DIM(_, data)                         \
  template class Wedge<DIM(data)>;                       \
  template bool operator==(const Wedge<DIM(data)>& lhs,  \
                           const Wedge<DIM(data)>& rhs); \
  template bool operator!=(const Wedge<DIM(data)>& lhs,  \
                           const Wedge<DIM(data)>& rhs);

#define INSTANTIATE_DTYPE(_, data)                                     \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data)> \
  Wedge<DIM(data)>::operator()(                                        \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const;  \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data),   \
                    Frame::NoFrame>                                    \
  Wedge<DIM(data)>::jacobian(                                          \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const;  \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data),   \
                    Frame::NoFrame>                                    \
  Wedge<DIM(data)>::inv_jacobian(                                      \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_DIM, (2, 3))
GENERATE_INSTANTIATIONS(INSTANTIATE_DTYPE, (2, 3),
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))

#undef DIM
#undef DTYPE
#undef INSTANTIATE_DIM
#undef INSTANTIATE_DTYPE
}  // namespace domain::CoordinateMaps
