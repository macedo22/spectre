// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#pragma once

#include <limits>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/TestHelpers.hpp"
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

  CHECK(get(L) == data);  // check LHS evaluated correctly

  // Test with Variables
  if constexpr (is_derived_of_vector_impl_v<DataType>) {
    Variables<tmpl::list<::Tags::TempTensor<0, Scalar<DataType>>,
                         ::Tags::TempTensor<1, Scalar<DataType>>>>
        vars(data.size(), std::numeric_limits<double>::signaling_NaN());

    Scalar<DataType>& R_temp =
        get<::Tags::TempTensor<0, Scalar<DataType>>>(vars);
    get(R_temp) = data;

    Scalar<DataType>& L_temp =
        get<::Tags::TempTensor<1, Scalar<DataType>>>(vars);
    call_evaluate<ReturnLhsTensor>(make_not_null(&L_temp), R());

    CHECK(get(R_temp) == data);  // check RHS wasn't modified
    CHECK(get(L_temp) == data);  // check LHS evaluated correctly
  }
}

}  // namespace TestHelpers::tenex
