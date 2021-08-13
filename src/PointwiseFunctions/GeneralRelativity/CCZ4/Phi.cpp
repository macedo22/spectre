// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/CCZ4/Phi.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace CCZ4 {
template <typename DataType>
void phi(const gsl::not_null<Scalar<DataType>*> phi,
         const Scalar<DataType>& det_spatial_metric) noexcept {
  get(*phi) = 1.0 / pow(get(det_spatial_metric), 1.0 / 6);
}

template <typename DataType>
Scalar<DataType> phi(const Scalar<DataType>& det_spatial_metric) noexcept {
  Scalar<DataType> phi{};
  ::CCZ4::phi(make_not_null(&phi), det_spatial_metric);
  return phi;
}
}  // namespace CCZ4

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                            \
  template void CCZ4::phi(                                              \
      const gsl::not_null<Scalar<DTYPE(data)>*> phi,                    \
      const Scalar<DTYPE(data)>& det_spatial_metric) noexcept;          \
  template Scalar<DTYPE(data)> CCZ4::phi(const Scalar < DTYPE(data >) & \
                                         det_spatial_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
