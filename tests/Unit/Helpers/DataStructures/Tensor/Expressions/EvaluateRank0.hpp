// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#pragma once

#include <type_traits>

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank2.hpp"  // TODO : remove after factoring out
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::tenex {
// TODO : update testing func docs

/// \ingroup TestingFrameworkGroup
/// \brief Test that evaluating a right hand side tensor expression containing a
/// single rank 0 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \param data the data being stored in the Tensors
template <bool ReturnLhsTensor, typename DataType>
void test_evaluate_rank_0(const DataType& data) {
  const Tensor<DataType> R{{{data}}};
  Scalar<DataType> L{};
  call_evaluate<ReturnLhsTensor>(make_not_null(&L), R());

  CHECK(get(L) == data);

  // Test with TempTensor for LHS tensor
  if constexpr (not std::is_same_v<DataType, double>) {
    Variables<tmpl::list<::Tags::TempTensor<1, Tensor<DataType>>>> L_var{
        data.size()};
    Tensor<DataType>& L_temp =
        get<::Tags::TempTensor<1, Tensor<DataType>>>(L_var);
    call_evaluate<false>(make_not_null(&L_temp), R());

    CHECK(get(L_temp) == data);
  }
}

}  // namespace TestHelpers::tenex
