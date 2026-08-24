// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class UniformCylindricalFlatEndcapShell.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain::CoordinateMaps {

/*!
 * \ingroup CoordinateMapsGroup
 *
 * \brief Map from a 3D right cylindrical shell to a volume that connects an
 * annular portion of a spherical surface with a flat annulus.
 *
 * \details Consider a sphere with center \f$C_1\f$ and radius \f$R_1\f$,
 * and a flat annulus in the \f$xy\f$ plane with center \f$C_2\f$, inner
 * radius \f$R_{2\mathrm{min}}\f$, and outer radius
 * \f$R_{2\mathrm{max}}\f$. The spherical annulus is bounded by the planes
 * \f$z=z_{\mathrm{P}1\mathrm{min}}\f$ and
 * \f$z=z_{\mathrm{P}1\mathrm{max}}\f$, where the `min` and `max` suffixes
 * refer to the polar angle, so
 * \f$z_{\mathrm{P}1\mathrm{min}}>z_{\mathrm{P}1\mathrm{max}}\f$.
 *
 * The source coordinates \f$(\bar{x},\bar{y},\bar{z})\f$ satisfy
 * \f$-1\leq\bar{z}\leq1\f$ and
 * \f$1\leq\sqrt{\bar{x}^2+\bar{y}^2}\leq2\f$. Define
 *
 * \f{align}
 * \bar\rho &= \sqrt{\bar{x}^2+\bar{y}^2},\\
 * s &= \bar\rho-1,\\
 * \lambda &= \frac{\bar z+1}{2},\\
 * \theta_1 &= \theta_{1\mathrm{min}}+
 *     s(\theta_{1\mathrm{max}}-\theta_{1\mathrm{min}}),\\
 * R_2(s) &= R_{2\mathrm{min}}+
 *     s(R_{2\mathrm{max}}-R_{2\mathrm{min}}),\\
 * \phi &= \operatorname{atan2}(\bar y,\bar x),
 * \f}
 * where
 *
 * \f{align}
 * \cos\theta_{1\mathrm{min}} &=(z_{\mathrm{P}1\mathrm{min}}-C_1^z)/R_1,\\
 * \cos\theta_{1\mathrm{max}} &=(z_{\mathrm{P}1\mathrm{max}}-C_1^z)/R_1.
 * \f}
 *
 * The map is
 *
 * \f{align}
 * x &= C_1^x+\lambda(C_2^x-C_1^x)+\cos\phi\left[
 *   (1-\lambda)R_1\sin\theta_1+\lambda R_2(s)\right],\\
 * y &= C_1^y+\lambda(C_2^y-C_1^y)+\sin\phi\left[
 *   (1-\lambda)R_1\sin\theta_1+\lambda R_2(s)\right],\\
 * z &= C_1^z+\lambda(C_2^z-C_1^z)
 *   +(1-\lambda)R_1\cos\theta_1.
 * \f}
 *
 * Thus, \f$\bar z=-1\f$ maps to a spherical annulus and
 * \f$\bar z=1\f$ maps to the flat annulus. The angular coordinate on the
 * sphere and the radius on the flat annulus are both uniform in \f$s\f$.
 * This is the annular analogue of `UniformCylindricalFlatEndcap`.
 *
 * The inverse is found by using the \f$z\f$ equation to express
 * \f$\lambda\f$ as a function of \f$s\f$ and numerically solving
 *
 * \f{align}
 * Q(s)={}&\left[x-C_1^x-\lambda(C_2^x-C_1^x)\right]^2
 *       +\left[y-C_1^y-\lambda(C_2^y-C_1^y)\right]^2\\
 * &-\left[(1-\lambda)R_1\sin\theta_1+\lambda R_2(s)\right]^2=0
 * \f}
 *
 * for \f$0\leq s\leq1\f$.
 */
class UniformCylindricalFlatEndcapShell {
 public:
  static constexpr size_t dim = 3;

  UniformCylindricalFlatEndcapShell(
      const std::array<double, 3>& center_one,
      const std::array<double, 3>& center_two, double radius_one,
      double radius_two_inner, double radius_two_outer,
      double z_plane_one_inner, double z_plane_one_outer);
  UniformCylindricalFlatEndcapShell() = default;
  ~UniformCylindricalFlatEndcapShell() = default;
  UniformCylindricalFlatEndcapShell(UniformCylindricalFlatEndcapShell&&) =
      default;
  UniformCylindricalFlatEndcapShell(
      const UniformCylindricalFlatEndcapShell&) = default;
  UniformCylindricalFlatEndcapShell& operator=(
      const UniformCylindricalFlatEndcapShell&) = default;
  UniformCylindricalFlatEndcapShell& operator=(
      UniformCylindricalFlatEndcapShell&&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords) const;

  /// The inverse function is only callable with doubles because it can fail
  /// for points outside the range of the map.
  std::optional<std::array<double, 3>> inverse(
      const std::array<double, 3>& target_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> jacobian(
      const std::array<T, 3>& source_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> inv_jacobian(
      const std::array<T, 3>& source_coords) const;

  // clang-tidy: google runtime references
  void pup(PUP::er& p);  // NOLINT

  static bool is_identity() { return false; }

  static constexpr bool supports_hessian{false};

 private:
  friend bool operator==(const UniformCylindricalFlatEndcapShell& lhs,
                         const UniformCylindricalFlatEndcapShell& rhs);

  std::array<double, 3> center_one_{};
  std::array<double, 3> center_two_{};
  double radius_one_{std::numeric_limits<double>::signaling_NaN()};
  double radius_two_inner_{std::numeric_limits<double>::signaling_NaN()};
  double radius_two_outer_{std::numeric_limits<double>::signaling_NaN()};
  double z_plane_one_inner_{std::numeric_limits<double>::signaling_NaN()};
  double z_plane_one_outer_{std::numeric_limits<double>::signaling_NaN()};
  double theta_one_inner_{std::numeric_limits<double>::signaling_NaN()};
  double theta_one_outer_{std::numeric_limits<double>::signaling_NaN()};
};

bool operator!=(const UniformCylindricalFlatEndcapShell& lhs,
                const UniformCylindricalFlatEndcapShell& rhs);

}  // namespace domain::CoordinateMaps
