// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/CCZ4/ConfSpatialMetric.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace CCZ4 {
template <size_t SpatialDim, typename Frame, typename DataType>
void conformal_spatial_metric(
    const gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*>
        conformal_spatial_metric,
    const Scalar<DataType>& phi_squared,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric) noexcept {
  ::TensorExpressions::evaluate<ti_i, ti_j>(
      conformal_spatial_metric, phi_squared() * spatial_metric(ti_i, ti_j));
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> conformal_spatial_metric(
    const Scalar<DataType>& phi_squared,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric) noexcept {
  tnsr::ii<DataType, SpatialDim, Frame> conformal_spatial_metric{};
  ::CCZ4::conformal_spatial_metric<SpatialDim, Frame, DataType>(
      make_not_null(&conformal_spatial_metric), phi_squared, spatial_metric);
  return conformal_spatial_metric;
}
}  // namespace CCZ4

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                              \
  template void CCZ4::conformal_spatial_metric(                           \
      const gsl::not_null<tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>*> \
          conformal_spatial_metric,                                       \
      const Scalar<DTYPE(data)>& phi_squared,                             \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                \
          spatial_metric) noexcept;                                       \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>                  \
  CCZ4::conformal_spatial_metric(                                         \
      const Scalar<DTYPE(data)>& phi_squared,                             \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                \
          spatial_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
