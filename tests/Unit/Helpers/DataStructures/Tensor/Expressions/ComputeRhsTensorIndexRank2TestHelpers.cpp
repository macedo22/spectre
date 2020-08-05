// Distributed under the MIT License.
// See LICENSE.txt for details.

//#include "Framework/TestingFramework.hpp"

#include "Helpers/DataStructures/Tensor/Expressions/ComputeRhsTensorIndexRank2TestHelpers.hpp"

#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TMPL.hpp"

// instead, this should iterate over
// 0 - SpatialDim1 (all i) and 0 - SpatialDim2 (all j)
// which means moving the for loops out of the test_rank_2
/*template <typename Datatype, typename Symmetry, typename IndexList,
          typename TensorIndexTypeA, typename TensorIndexTypeB>
void test_compute_rhs_tensor_index_rank_2_core_impl(
    const TensorIndexTypeA& tensor_index_type_a,
    const TensorIndexTypeB& tensor_index_type_b, const size_t& spatial_dim_a,
    const size_t& spatial_dim_b) {
  Tensor<Datatype, Symmetry, IndexList> rhs_tensor{};

  auto rhs_tensor_expr = rhs_tensor(tensor_index_type_a, tensor_index_type_b);

  std::array<size_t, 2> index_order_ab = {TensorIndexTypeA::value,
                                          TensorIndexTypeB::value};
  std::array<size_t, 2> index_order_ba = {TensorIndexTypeB::value,
                                          TensorIndexTypeA::value};

  for (size_t i = 0; i < spatial_dim_a; i++) {
    for (size_t j = 0; j < spatial_dim_b; j++) {
      const std::array<size_t, 2> ij = {i, j};
      const std::array<size_t, 2> ji = {j, i};

      CHECK(rhs_tensor_expr.template compute_rhs_tensor_index<2>(
                index_order_ab, index_order_ab, ij) == ij);
      CHECK(rhs_tensor_expr.template compute_rhs_tensor_index<2>(
                index_order_ba, index_order_ab, ij) == ji);
    }
  }
}*/
