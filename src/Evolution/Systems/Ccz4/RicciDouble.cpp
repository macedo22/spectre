// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/Ricci.hpp"

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

#include "Ricci.cpp"

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                              \
  template void Ccz4::spatial_ricci_tensor(                               \
      const gsl::not_null<tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>*> \
          result,                                                         \
      const gsl::not_null<tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>*> \
          buffer,                                                         \
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
      const tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>& d_field_p);    \
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
      const tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>& d_field_p);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial),
                        (double))

#undef INSTANTIATE
#undef DTYPE
#undef FRAME
#undef DIM
