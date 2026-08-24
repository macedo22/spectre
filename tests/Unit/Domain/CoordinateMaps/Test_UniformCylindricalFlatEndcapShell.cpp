// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>

#include "Domain/CoordinateMaps/UniformCylindricalFlatEndcap.hpp"
#include "Domain/CoordinateMaps/UniformCylindricalFlatEndcapShell.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"

namespace domain {
namespace {

void test_uniform_cylindrical_flat_endcap_shell() {
  const std::array<double, 3> center_one{{0.1, -0.2, 0.3}};
  const std::array<double, 3> center_two{{0.15, -0.16, 6.3}};
  constexpr double radius_one = 2.0;
  constexpr double theta_inner = 0.12 * M_PI;
  constexpr double theta_outer = 0.25 * M_PI;
  constexpr double radius_two_inner = 0.9;
  constexpr double radius_two_outer = 2.2;
  const double z_plane_one_inner =
      center_one[2] + radius_one * cos(theta_inner);
  const double z_plane_one_outer =
      center_one[2] + radius_one * cos(theta_outer);

  const CoordinateMaps::UniformCylindricalFlatEndcapShell map{
      center_one,          center_two,          radius_one,
      radius_two_inner,    radius_two_outer,    z_plane_one_inner,
      z_plane_one_outer};
  test_suite_for_map_on_cylinder(map, 1.0, 2.0, true, true);

  // The inner surface of the shell map must agree pointwise with the outer
  // surface of the filled endcap map constructed from the inner parameters.
  const CoordinateMaps::UniformCylindricalFlatEndcap filled_map{
      center_one, center_two, radius_one, radius_two_inner,
      z_plane_one_inner};
  for (const double phi : {0.0, 0.7, 2.4}) {
    for (const double zbar : {-1.0, -0.3, 1.0}) {
      const std::array<double, 3> source{{cos(phi), sin(phi), zbar}};
      CHECK_ITERABLE_APPROX(map(source), filled_map(source));
    }
  }

  CHECK_FALSE(map.inverse({{center_one[0], center_one[1],
                            z_plane_one_outer - 0.1}})
                  .has_value());
  CHECK_FALSE(map.inverse({{center_two[0], center_two[1],
                            center_two[2] + 0.1}})
                  .has_value());
  CHECK_FALSE(map.inverse({{center_one[0], center_one[1],
                            center_one[2] + 0.99 * radius_one}})
                  .has_value());
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Domain.CoordinateMaps.UniformCylindricalFlatEndcapShell",
    "[Domain][Unit]") {
  test_uniform_cylindrical_flat_endcap_shell();
  CHECK(not CoordinateMaps::UniformCylindricalFlatEndcapShell{}.is_identity());
}

}  // namespace domain
