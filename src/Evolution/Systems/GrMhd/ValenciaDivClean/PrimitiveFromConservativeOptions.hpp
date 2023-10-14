// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace grmhd::ValenciaDivClean {

/// Options to be passed to the Con2Prim algorithm.
/// Currently, we simply set a threshold for tildeD
/// below which the inversion is not performed and
/// the density is set to atmosphere values.
class PrimitiveFromConservativeOptions {
 public:
  struct CutoffDForInversion {
    static std::string name() { return "CutoffDForInversion"; }
    Options::String help;
    using type = double;
    static type lower_bound() { return 0.0; }
  };

  struct DensityWhenSkippingInversion {
    static std::string name() { return "DensityWhenSkippingInversion"; }
    Options::String help;
    using type = double;
    static type lower_bound() { return 0.0; }
  };

  using options = tmpl::list<CutoffDForInversion, DensityWhenSkippingInversion>;

  Options::String help;

  PrimitiveFromConservativeOptions() = default;

  PrimitiveFromConservativeOptions(
      const double cutoff_d_for_inversion,
      const double density_when_skipping_inversion);

  void pup(PUP::er& p);

  double cutoff_d_for_inversion() const { return cutoff_d_for_inversion_; }
  double density_when_skipping_inversion() const {
    return density_when_skipping_inversion_;
  }

 private:
  friend bool operator==(const PrimitiveFromConservativeOptions& lhs,
                         const PrimitiveFromConservativeOptions& rhs);

  double cutoff_d_for_inversion_ = std::numeric_limits<double>::signaling_NaN();
  double density_when_skipping_inversion_ =
      std::numeric_limits<double>::signaling_NaN();
};

bool operator!=(const PrimitiveFromConservativeOptions& lhs,
                const PrimitiveFromConservativeOptions& rhs);

}  // namespace grmhd::ValenciaDivClean
