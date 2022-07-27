// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Options/Protocols/FactoryCreation.hpp"

namespace test {
struct idk;

// template <typename AhA, typename AhB, typename control_systems, size_t volume_dim, typename interpolator_source_vars,
//           typename observe_fields, typename non_tensor_compute_tags, typename system, bool local_time_stepping>
struct factory_creation;
// struct factory_creation : tt::ConformsTo<Options::protocols::FactoryCreation> {
//     struct factory_classes;
//     // using factory_classes = impl;
// };
}  // namespace test
