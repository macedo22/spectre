// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CCZ4/TimeDerivative.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace CCZ4 {
template <size_t Dim, typename DataType>
void TimeDerivative<Dim, DataType>::apply(
    const gsl::not_null<tnsr::ij<DataType, Dim>*> dt_conf_spatial_metric,
    const gsl::not_null<Scalar<DataType>*> dt_ln_lapse,
    const gsl::not_null<tnsr::I<DataType, Dim>*> dt_shift,
    const gsl::not_null<Scalar<DataType>*> det_conf_spatial_metric,
    const gsl::not_null<Scalar<DataType>*> trace_A_tilde,
    const tnsr::ii<DataType, Dim>& conf_spatial_metric,
    const tnsr::I<DataType, Dim>& shift,
    const Scalar<DataType>& trace_extrinsic_curvature,
    const Scalar<DataType>& K_0, const tnsr::i<DataType, Dim>& A,
    const tnsr::ijk<DataType, Dim>& D, const tnsr::iJ<DataType, Dim>& B,
    const Scalar<DataType>& lapse, const Scalar<DataType>& g,
    const Scalar<DataType>& theta, const double c,
    const tnsr::ij<DataType, Dim>& A_tilde, const double relaxation_time,
    const double s, const double f, const tnsr::I<DataType, Dim>& b) noexcept {
  // dt_conf_spatial_metric: time derivative of the conformal spatial metric
  ::TensorExpressions::evaluate<ti_i, ti_j>(
      dt_conf_spatial_metric,
      2.0 * shift(ti_K) * D(ti_k, ti_i, ti_j) +
          conf_spatial_metric(ti_k, ti_i) * B(ti_j, ti_K) +
          conf_spatial_metric(ti_k, ti_j) * B(ti_i, ti_K) -
          (2.0 / 3) * conf_spatial_metric(ti_i, ti_j) * B(ti_k, ti_K) -
          2.0 * lapse() *
              (A_tilde(ti_i, ti_j) -
               (1.0 / 3) *
                   (conf_spatial_metric(ti_i, ti_j) * (*trace_A_tilde)())) -
          (1.0 / relaxation_time) * ((*det_conf_spatial_metric)() - 1) *
              conf_spatial_metric(ti_i, ti_j));

  // dt_ln_lapse: time derivative of the natural log of the lapse
  ::TensorExpressions::evaluate(
      dt_ln_lapse, shift(ti_K) * A(ti_k) - lapse() * g() *
                                               (trace_extrinsic_curvature() -
                                                K_0() - 2.0 * theta() * c));

  // dt_shift: time derivative of the shift
  ::TensorExpressions::evaluate<ti_I>(
      dt_shift, s * shift(ti_K) * B(ti_k, ti_I) + s * f * b(ti_I));
}
}  // namespace CCZ4

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DATATYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATE(_, data) \
  template struct CCZ4::TimeDerivative<DIM(data), DATATYPE(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector))

#undef INSTANTIATE
#undef DATATYPE
#undef DIM
