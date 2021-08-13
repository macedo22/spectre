// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/CCZ4/PhiSquared.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace CCZ4 {
template <typename DataType>
void phi_squared(const gsl::not_null<Scalar<DataType>*> phi_squared,
                 const Scalar<DataType>& phi) noexcept {
  get(*phi_squared) = get(phi) * get(phi);
}

template <typename DataType>
Scalar<DataType> phi_squared(const Scalar<DataType>& phi) noexcept {
  Scalar<DataType> phi_squared{};
  ::CCZ4::phi_squared(make_not_null(&phi_squared), phi);
  return phi_squared;
}
}  // namespace CCZ4

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                 \
  template void CCZ4::phi_squared(                           \
      const gsl::not_null<Scalar<DTYPE(data)>*> phi_squared, \
      const Scalar<DTYPE(data)>& phi) noexcept;              \
  template Scalar<DTYPE(data)> CCZ4::phi_squared(            \
      const Scalar<DTYPE(data)>& phi) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
