// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"

namespace CCZ4 {

/// \brief Tags for the CCZ4 formulation of Einstein equations
namespace Tags {
template <typename DataType>
struct Phi;
template <typename DataType>
struct PhiSquared;
template <size_t SpatialDim, typename Frame, typename DataType>
struct ConfSpatialMetric;
template <size_t SpatialDim, typename Frame, typename DataType>
struct ExtrinsicCurvature;
template <typename DataType>
struct TraceExtrinsicCurvature;
template <size_t SpatialDim, typename Frame, typename DataType>
struct ATilde;
}  // namespace Tags

/// \brief Input option tags for the generalized harmonic evolution system
namespace OptionTags {
struct Group;
}  // namespace OptionTags
}  // namespace CCZ4
