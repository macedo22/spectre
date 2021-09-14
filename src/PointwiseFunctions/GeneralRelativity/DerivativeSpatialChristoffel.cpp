// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>

#include "PointwiseFunctions/GeneralRelativity/DerivativeSpatialChristoffel.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace gr {
template <size_t Dim, typename Frame, typename DataType>
void deriv_spatial_christoffel_second_kind(
    const gsl::not_null<tnsr::iJkk<DataType, Dim, Frame>*> d_christoffel,
    const tnsr::iJJ<DataType, Dim, Frame>& d_inverse_spatial_metric,
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& d_spatial_metric,
    const Tensor<DataType, Symmetry<2, 2, 1, 1>,
                 index_list<SpatialIndex<Dim, UpLo::Lo, Frame>,
                            SpatialIndex<Dim, UpLo::Lo, Frame>,
                            SpatialIndex<Dim, UpLo::Lo, Frame>,
                            SpatialIndex<Dim, UpLo::Lo, Frame>>>&
        d2_spatial_metric) noexcept {
  destructive_resize_components(d_christoffel,
                                get_size(get<0, 0>(inverse_spatial_metric)));

  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      for (size_t k = 0; k < Dim; ++k) {
        for (size_t m = 0; m < Dim; ++m) {
          for (size_t l = 0; l < Dim; ++l) {
            (*d_christoffel).get(k, m, i, j) =
                (2.0 * inverse_spatial_metric.get(m, l) *
                 (d2_spatial_metric.get(k, i, j, l) +
                  d2_spatial_metric.get(k, j, i, l) -
                  d2_spatial_metric.get(k, l, i, j))) -
                d_inverse_spatial_metric.get(k, m, l) *
                    (d_spatial_metric.get(i, j, l) +
                     d_spatial_metric.get(j, i, l) +
                     d_spatial_metric.get(l, i, j));
          }
        }
      }
    }
  }
}

template <size_t Dim, typename Frame, typename DataType>
tnsr::iJkk<DataType, Dim, Frame> deriv_spatial_christoffel_second_kind(
    const tnsr::iJJ<DataType, Dim, Frame>& d_inverse_spatial_metric,
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& d_spatial_metric,
    const Tensor<DataType, Symmetry<2, 2, 1, 1>,
                 index_list<SpatialIndex<Dim, UpLo::Lo, Frame>,
                            SpatialIndex<Dim, UpLo::Lo, Frame>,
                            SpatialIndex<Dim, UpLo::Lo, Frame>,
                            SpatialIndex<Dim, UpLo::Lo, Frame>>>&
        d2_spatial_metric) noexcept {
  tnsr::iJkk<DataType, Dim, Frame> d_christoffel{};
  deriv_spatial_christoffel_second_kind(
      make_not_null(&d_christoffel), d_inverse_spatial_metric,
      inverse_spatial_metric, d_spatial_metric, d2_spatial_metric);
  return d_christoffel;
}
}  // namespace gr

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                  \
  template void gr::deriv_spatial_christoffel_second_kind(                    \
      const gsl::not_null<tnsr::iJkk<DTYPE(data), DIM(data), FRAME(data)>*>   \
          d_christoffel,                                                      \
      const tnsr::iJJ<DTYPE(data), DIM(data), FRAME(data)>&                   \
          d_inverse_spatial_metric,                                           \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>& d_spatial_metric, \
      const Tensor<                                                           \
          DTYPE(data), Symmetry<2, 2, 1, 1>,                                  \
          index_list<SpatialIndex<DIM(data), UpLo::Lo, FRAME(data)>,          \
                     SpatialIndex<DIM(data), UpLo::Lo, FRAME(data)>,          \
                     SpatialIndex<DIM(data), UpLo::Lo, FRAME(data)>,          \
                     SpatialIndex<DIM(data), UpLo::Lo, FRAME(data)>>>&        \
          d2_spatial_metric) noexcept;                                        \
  template tnsr::iJkk<DTYPE(data), DIM(data), FRAME(data)>                    \
  gr::deriv_spatial_christoffel_second_kind(                                  \
      const tnsr::iJJ<DTYPE(data), DIM(data), FRAME(data)>&                   \
          d_inverse_spatial_metric,                                           \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>& d_spatial_metric, \
      const Tensor<                                                           \
          DTYPE(data), Symmetry<2, 2, 1, 1>,                                  \
          index_list<SpatialIndex<DIM(data), UpLo::Lo, FRAME(data)>,          \
                     SpatialIndex<DIM(data), UpLo::Lo, FRAME(data)>,          \
                     SpatialIndex<DIM(data), UpLo::Lo, FRAME(data)>,          \
                     SpatialIndex<DIM(data), UpLo::Lo, FRAME(data)>>>&        \
          d2_spatial_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial),
                        (double, DataVector))

#undef DTYPE
#undef FRAME
#undef DIM
#undef INSTANTIATE
