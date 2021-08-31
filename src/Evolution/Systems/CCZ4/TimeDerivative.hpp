// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;

namespace PUP {
class er;
}  // namespace PUP

namespace gsl {
template <class T>
class not_null;
}  // namespace gsl

template <typename, typename, typename>
class Tensor;
/// \endcond

namespace CCZ4 {
template <size_t Dim, typename DataType>
struct TimeDerivative {
 public:
  static void apply(
      const gsl::not_null<tnsr::ij<DataType, Dim>*> dt_conf_spatial_metric,
      const gsl::not_null<Scalar<DataType>*> det_conf_spatial_metric,
      const gsl::not_null<Scalar<DataType>*> trace_A_tilde,
      const tnsr::ii<DataType, Dim>& conf_spatial_metric,
      const tnsr::I<DataType, Dim>& shift, const tnsr::ijk<DataType, Dim>& D,
      const tnsr::iJ<DataType, Dim>& B, const Scalar<DataType>& lapse,
      const tnsr::ij<DataType, Dim>& A_tilde,
      const double relaxation_time) noexcept;
};
}  // namespace CCZ4
