// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/Ricci.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4 {
template <size_t Dim, typename Frame, typename DataType>
void spatial_ricci_tensor(
    const gsl::not_null<tnsr::ii<DataType, Dim, Frame>*> result,
    const tnsr::Ijj<DataType, Dim, Frame>& christoffel_second_kind,
    const tnsr::iJkk<DataType, Dim, Frame>& d_conformal_christoffel_second_kind,
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& field_d,
    const tnsr::iJJ<DataType, Dim, Frame>& field_d_up,
    const tnsr::i<DataType, Dim, Frame>& field_p,
    const tnsr::ij<DataType, Dim, Frame>& d_field_p) noexcept {
  destructive_resize_components(result,
                                get_size(get<0, 0>(conformal_spatial_metric)));
  for (auto& component : (*result)) {
    component = 0.0;
  }

  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = i; j < Dim; j++) {
      for (size_t m = 0; m < Dim; m++) {
        // Add first terms of \partial_m \Gamma^m{}_{ij} and
        // -\partial_j \Gamma^m{}_{im}
        result->get(i, j) +=
            d_conformal_christoffel_second_kind.get(m, m, i, j) -
            d_conformal_christoffel_second_kind.get(j, m, i, m);
        for (size_t l = 0; l < Dim; l++) {
          result->get(i, j) +=
              // Add terms of \partial_m \Gamma^m{}_{ij} and
              // -\partial_j \Gamma^m{}_{im} that have a coefficient of 2
              2.0 * ((field_d_up.get(m, m, l) *
                      (conformal_spatial_metric.get(j, l) * field_p.get(i) +
                       conformal_spatial_metric.get(i, l) * field_p.get(j) -
                       conformal_spatial_metric.get(i, j) * field_p.get(l))) -
                     inverse_conformal_spatial_metric.get(m, l) *
                         (field_d.get(m, j, l) * field_p.get(i) +
                          field_d.get(m, i, l) * field_p.get(j) -
                          field_d.get(m, i, j) * field_p.get(l)) -
                     (field_d_up.get(j, m, l) *
                      (conformal_spatial_metric.get(m, l) * field_p.get(i) +
                       conformal_spatial_metric.get(i, l) * field_p.get(m) -
                       conformal_spatial_metric.get(i, m) * field_p.get(l))) +
                     inverse_conformal_spatial_metric.get(m, l) *
                         (field_d.get(j, m, l) * field_p.get(i) +
                          field_d.get(j, i, l) * field_p.get(m) -
                          field_d.get(j, i, m) * field_p.get(l))) -
              // Add terms of \partial_m \Gamma^m{}_{ij} and
              // -\partial_j \Gamma^m{}_{im} that have a coefficient of 1/2
              0.5 *
                  (inverse_conformal_spatial_metric.get(m, l) *
                   (conformal_spatial_metric.get(j, l) * d_field_p.get(m, i) +
                    conformal_spatial_metric.get(j, l) * d_field_p.get(i, m) +
                    conformal_spatial_metric.get(i, l) * d_field_p.get(m, j) +
                    conformal_spatial_metric.get(i, l) * d_field_p.get(j, m) -
                    conformal_spatial_metric.get(i, j) * d_field_p.get(m, l) -
                    conformal_spatial_metric.get(i, j) * d_field_p.get(l, m) -
                    conformal_spatial_metric.get(m, l) * d_field_p.get(j, i) -
                    conformal_spatial_metric.get(m, l) * d_field_p.get(i, j) -
                    conformal_spatial_metric.get(i, l) * d_field_p.get(j, m) -
                    conformal_spatial_metric.get(i, l) * d_field_p.get(m, j) +
                    conformal_spatial_metric.get(i, m) * d_field_p.get(j, l) +
                    conformal_spatial_metric.get(i, m) * d_field_p.get(l, j))) +
              // Add last two terms for R_{ij}
              christoffel_second_kind.get(l, i, j) *
                  christoffel_second_kind.get(m, l, m) -
              christoffel_second_kind.get(l, i, m) *
                  christoffel_second_kind.get(m, l, j);
        }
      }
    }
  }
}

template <size_t Dim, typename Frame, typename DataType>
tnsr::ii<DataType, Dim, Frame> spatial_ricci_tensor(
    const tnsr::Ijj<DataType, Dim, Frame>& christoffel_second_kind,
    const tnsr::iJkk<DataType, Dim, Frame>& d_conformal_christoffel_second_kind,
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& field_d,
    const tnsr::iJJ<DataType, Dim, Frame>& field_d_up,
    const tnsr::i<DataType, Dim, Frame>& field_p,
    const tnsr::ij<DataType, Dim, Frame>& d_field_p) noexcept {
  tnsr::ii<DataType, Dim, Frame> result{};
  spatial_ricci_tensor(make_not_null(&result), christoffel_second_kind,
                       d_conformal_christoffel_second_kind,
                       conformal_spatial_metric,
                       inverse_conformal_spatial_metric, field_d, field_d_up,
                       field_p, d_field_p);
  return result;
}
}  // namespace Ccz4

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                              \
  template void Ccz4::spatial_ricci_tensor(                               \
      const gsl::not_null<tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>*> \
          result,                                                         \
      const tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>&               \
          christoffel_second_kind,                                        \
      const tnsr::iJkk<DTYPE(data), DIM(data), FRAME(data)>&              \
          d_conformal_christoffel_second_kind,                            \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                \
          conformal_spatial_metric,                                       \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                \
          inverse_conformal_spatial_metric,                               \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>& field_d,      \
      const tnsr::iJJ<DTYPE(data), DIM(data), FRAME(data)>& field_d_up,   \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& field_p,        \
      const tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>&                \
          d_field_p) noexcept;                                            \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>                  \
  Ccz4::spatial_ricci_tensor(                                             \
      const tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>&               \
          christoffel_second_kind,                                        \
      const tnsr::iJkk<DTYPE(data), DIM(data), FRAME(data)>&              \
          d_conformal_christoffel_second_kind,                            \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                \
          conformal_spatial_metric,                                       \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                \
          inverse_conformal_spatial_metric,                               \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>& field_d,      \
      const tnsr::iJJ<DTYPE(data), DIM(data), FRAME(data)>& field_d_up,   \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& field_p,        \
      const tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>&                \
          d_field_p) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial),
                        (double, DataVector))

#undef INSTANTIATE
#undef DTYPE
#undef FRAME
#undef DIM
