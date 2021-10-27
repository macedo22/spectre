// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <climits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace {
using R_type = Tensor<DataVector, Symmetry<4, 3, 2, 1>,
                      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>;

using S_type = Tensor<DataVector, Symmetry<4, 3, 2, 1>,
                      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>;

template <size_t Count>
SPECTRE_ALWAYS_INLINE decltype(auto) get(const R_type& R, const S_type& S,
                                         std::array<size_t, 4>& multi_index) {
  if constexpr (Count == 256) {
    return R.get(multi_index) * S.get(multi_index);
  } else {
    return R.get(multi_index) * S.get(multi_index) +
           get<Count + 1>(R, S, multi_index);
  }
}

SPECTRE_ALWAYS_INLINE decltype(auto) get(const R_type& R, const S_type& S) {
  std::array<size_t, 4> multi_index{0, 0, 0, 0};
  constexpr size_t count = 1;
  return R.get(multi_index) * S.get(multi_index) +
         get<count + 1>(R, S, multi_index);
}

template <size_t NumContractedIndices>
struct InnerProduct {
  static constexpr size_t dim = 4;
  static constexpr size_t num_contracted_indices = NumContractedIndices;
  static constexpr size_t num_uncontracted_indices = NumContractedIndices + 2;

  // Get R or S component from leaves
  template <typename T>
  SPECTRE_ALWAYS_INLINE decltype(auto) get(const T& t,
                                           std::array<size_t, 4> multi_index) {
    return t.get(multi_index);
  }

  template <size_t Iteration>
  SPECTRE_ALWAYS_INLINE decltype(auto) compute_contraction(
      const R_type& R, const S_type& S,
      std::array<size_t, num_uncontracted_indices> uncontracted_multi_index) {
    if constexpr (Iteration == dim - 1) {
      return InnerProduct<num_uncontracted_indices>{}.get(
          R, S, uncontracted_multi_index);
    } else {
      return InnerProduct<num_uncontracted_indices>{}.get(
                 R, S, uncontracted_multi_index) +
             compute_contraction<Iteration + 1>(R, S, uncontracted_multi_index);
    }
  }

  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const R_type& R, const S_type& S,
      const std::array<size_t, num_contracted_indices>&
          contracted_multi_index) {
    if constexpr (num_contracted_indices == 8) {
      // return product
      const std::array<size_t, 4> multi_index{
          contracted_multi_index[0], contracted_multi_index[1],
          contracted_multi_index[2], contracted_multi_index[3]};
      return get(R, multi_index) * get(S, multi_index);
    } else {
      // sum over dim for this pair of indices
      auto uncontracted_multi_index =
          make_array<num_uncontracted_indices, size_t>(0);
      for (size_t i = 0; i < num_contracted_indices; i++) {
        uncontracted_multi_index[i] = contracted_multi_index[i];
      }
      return compute_contraction<0>(R, S, uncontracted_multi_index);
    }
  }
};

decltype(auto) evaluate(const R_type& R, const S_type& S) {
  constexpr std::array<size_t, 0> contracted_multi_index{};
  return InnerProduct<0>{}.get(R, S, contracted_multi_index);
}

template <typename Generator>
void test_large_datavector_expression(const gsl::not_null<Generator*> generator,
                                      const DataVector& used_for_size) {
  std::uniform_real_distribution<> distribution(-1.0, 1.0);
  const auto R =
      make_with_random_values<R_type>(generator, distribution, used_for_size);
  const auto S =
      make_with_random_values<S_type>(generator, distribution, used_for_size);

  // Rank 4 x Rank 4 inner product
  // 3D, contract over spatial dimensions
  //
  // Compiled with clang-10 compile_commands.json command. See
  // compile_command.txt in this directory
  //
  // real    0m16.151s
  // user    0m15.860s
  // sys     0m0.290s
  // const Scalar<DataVector> T = TensorExpressions::evaluate(
  //     R(ti_I, ti_J, ti_K, ti_L) * S(ti_i, ti_j, ti_k, ti_l));

  // Rank 4 x Rank 4 inner product
  // 3D, contract over all 4 dimensions
  //
  // Compiled with clang-10 compile_commands.json command. See
  // compile_command.txt in this directory
  //
  // real    5m17.453s
  // user    5m16.403s
  // sys     0m1.044s
  // const Scalar<DataVector> L = TensorExpressions::evaluate(
  //     R(ti_A, ti_B, ti_C, ti_D) * S(ti_d, ti_c, ti_b, ti_a));

  // Rank 4 x Rank 4 inner product
  // 3D, contract over all 4 dimensions
  //
  // Compiled with clang-10 compile_commands.json command. See
  // compile_command.txt in this directory
  //
  // real    1m6.114s
  // user    1m5.108s
  // sys     0m1.004s
  // const DataVector result = get(R, S);
  // (void)result;

  // Rank 4 x Rank 4 inner product
  // 3D, contract over all 4 dimensions
  //
  // Compiled with clang-10 compile_commands.json command. See
  // compile_command.txt in this directory
  //
  // real    0m48.824s
  // user    0m48.239s
  // sys     0m0.584s
  // const DataVector result = evaluate(R, S);
  // CHECK_ITERABLE_APPROX(result, 256 * R.get(0, 0, 0, 0) * S.get(0, 0, 0, 0));

  // Rank 4 x Rank 4 inner product
  // 3D, contract over all 4 dimensions
  //
  // Compiled with clang-10 compile_commands.json command. See
  // compile_command.txt in this directory
  //
  // real    1m27.180s
  // user    1m25.824s
  // sys     0m1.341s
  // const DataVector result = R.get(0, 0, 0, 0) * S.get(0, 0, 0, 0) +
  //                           R.get(0, 0, 0, 1) * S.get(1, 0, 0, 0) +
  //                           R.get(0, 0, 0, 2) * S.get(2, 0, 0, 0) +
  //                           R.get(0, 0, 0, 3) * S.get(3, 0, 0, 0) +
  //                           R.get(0, 0, 1, 0) * S.get(0, 1, 0, 0) +
  //                           R.get(0, 0, 1, 1) * S.get(1, 1, 0, 0) +
  //                           R.get(0, 0, 1, 2) * S.get(2, 1, 0, 0) +
  //                           R.get(0, 0, 1, 3) * S.get(3, 1, 0, 0) +
  //                           R.get(0, 0, 2, 0) * S.get(0, 2, 0, 0) +
  //                           R.get(0, 0, 2, 1) * S.get(1, 2, 0, 0) +
  //                           R.get(0, 0, 2, 2) * S.get(2, 2, 0, 0) +
  //                           R.get(0, 0, 2, 3) * S.get(3, 2, 0, 0) +
  //                           R.get(0, 0, 3, 0) * S.get(0, 3, 0, 0) +
  //                           R.get(0, 0, 3, 1) * S.get(1, 3, 0, 0) +
  //                           R.get(0, 0, 3, 2) * S.get(2, 3, 0, 0) +
  //                           R.get(0, 0, 3, 3) * S.get(3, 3, 0, 0) +
  //                           R.get(0, 1, 0, 0) * S.get(0, 0, 1, 0) +
  //                           R.get(0, 1, 0, 1) * S.get(1, 0, 1, 0) +
  //                           R.get(0, 1, 0, 2) * S.get(2, 0, 1, 0) +
  //                           R.get(0, 1, 0, 3) * S.get(3, 0, 1, 0) +
  //                           R.get(0, 1, 1, 0) * S.get(0, 1, 1, 0) +
  //                           R.get(0, 1, 1, 1) * S.get(1, 1, 1, 0) +
  //                           R.get(0, 1, 1, 2) * S.get(2, 1, 1, 0) +
  //                           R.get(0, 1, 1, 3) * S.get(3, 1, 1, 0) +
  //                           R.get(0, 1, 2, 0) * S.get(0, 2, 1, 0) +
  //                           R.get(0, 1, 2, 1) * S.get(1, 2, 1, 0) +
  //                           R.get(0, 1, 2, 2) * S.get(2, 2, 1, 0) +
  //                           R.get(0, 1, 2, 3) * S.get(3, 2, 1, 0) +
  //                           R.get(0, 1, 3, 0) * S.get(0, 3, 1, 0) +
  //                           R.get(0, 1, 3, 1) * S.get(1, 3, 1, 0) +
  //                           R.get(0, 1, 3, 2) * S.get(2, 3, 1, 0) +
  //                           R.get(0, 1, 3, 3) * S.get(3, 3, 1, 0) +
  //                           R.get(0, 2, 0, 0) * S.get(0, 0, 2, 0) +
  //                           R.get(0, 2, 0, 1) * S.get(1, 0, 2, 0) +
  //                           R.get(0, 2, 0, 2) * S.get(2, 0, 2, 0) +
  //                           R.get(0, 2, 0, 3) * S.get(3, 0, 2, 0) +
  //                           R.get(0, 2, 1, 0) * S.get(0, 1, 2, 0) +
  //                           R.get(0, 2, 1, 1) * S.get(1, 1, 2, 0) +
  //                           R.get(0, 2, 1, 2) * S.get(2, 1, 2, 0) +
  //                           R.get(0, 2, 1, 3) * S.get(3, 1, 2, 0) +
  //                           R.get(0, 2, 2, 0) * S.get(0, 2, 2, 0) +
  //                           R.get(0, 2, 2, 1) * S.get(1, 2, 2, 0) +
  //                           R.get(0, 2, 2, 2) * S.get(2, 2, 2, 0) +
  //                           R.get(0, 2, 2, 3) * S.get(3, 2, 2, 0) +
  //                           R.get(0, 2, 3, 0) * S.get(0, 3, 2, 0) +
  //                           R.get(0, 2, 3, 1) * S.get(1, 3, 2, 0) +
  //                           R.get(0, 2, 3, 2) * S.get(2, 3, 2, 0) +
  //                           R.get(0, 2, 3, 3) * S.get(3, 3, 2, 0) +
  //                           R.get(0, 3, 0, 0) * S.get(0, 0, 3, 0) +
  //                           R.get(0, 3, 0, 1) * S.get(1, 0, 3, 0) +
  //                           R.get(0, 3, 0, 2) * S.get(2, 0, 3, 0) +
  //                           R.get(0, 3, 0, 3) * S.get(3, 0, 3, 0) +
  //                           R.get(0, 3, 1, 0) * S.get(0, 1, 3, 0) +
  //                           R.get(0, 3, 1, 1) * S.get(1, 1, 3, 0) +
  //                           R.get(0, 3, 1, 2) * S.get(2, 1, 3, 0) +
  //                           R.get(0, 3, 1, 3) * S.get(3, 1, 3, 0) +
  //                           R.get(0, 3, 2, 0) * S.get(0, 2, 3, 0) +
  //                           R.get(0, 3, 2, 1) * S.get(1, 2, 3, 0) +
  //                           R.get(0, 3, 2, 2) * S.get(2, 2, 3, 0) +
  //                           R.get(0, 3, 2, 3) * S.get(3, 2, 3, 0) +
  //                           R.get(0, 3, 3, 0) * S.get(0, 3, 3, 0) +
  //                           R.get(0, 3, 3, 1) * S.get(1, 3, 3, 0) +
  //                           R.get(0, 3, 3, 2) * S.get(2, 3, 3, 0) +
  //                           R.get(0, 3, 3, 3) * S.get(3, 3, 3, 0) +
  //                           R.get(1, 0, 0, 0) * S.get(0, 0, 0, 1) +
  //                           R.get(1, 0, 0, 1) * S.get(1, 0, 0, 1) +
  //                           R.get(1, 0, 0, 2) * S.get(2, 0, 0, 1) +
  //                           R.get(1, 0, 0, 3) * S.get(3, 0, 0, 1) +
  //                           R.get(1, 0, 1, 0) * S.get(0, 1, 0, 1) +
  //                           R.get(1, 0, 1, 1) * S.get(1, 1, 0, 1) +
  //                           R.get(1, 0, 1, 2) * S.get(2, 1, 0, 1) +
  //                           R.get(1, 0, 1, 3) * S.get(3, 1, 0, 1) +
  //                           R.get(1, 0, 2, 0) * S.get(0, 2, 0, 1) +
  //                           R.get(1, 0, 2, 1) * S.get(1, 2, 0, 1) +
  //                           R.get(1, 0, 2, 2) * S.get(2, 2, 0, 1) +
  //                           R.get(1, 0, 2, 3) * S.get(3, 2, 0, 1) +
  //                           R.get(1, 0, 3, 0) * S.get(0, 3, 0, 1) +
  //                           R.get(1, 0, 3, 1) * S.get(1, 3, 0, 1) +
  //                           R.get(1, 0, 3, 2) * S.get(2, 3, 0, 1) +
  //                           R.get(1, 0, 3, 3) * S.get(3, 3, 0, 1) +
  //                           R.get(1, 1, 0, 0) * S.get(0, 0, 1, 1) +
  //                           R.get(1, 1, 0, 1) * S.get(1, 0, 1, 1) +
  //                           R.get(1, 1, 0, 2) * S.get(2, 0, 1, 1) +
  //                           R.get(1, 1, 0, 3) * S.get(3, 0, 1, 1) +
  //                           R.get(1, 1, 1, 0) * S.get(0, 1, 1, 1) +
  //                           R.get(1, 1, 1, 1) * S.get(1, 1, 1, 1) +
  //                           R.get(1, 1, 1, 2) * S.get(2, 1, 1, 1) +
  //                           R.get(1, 1, 1, 3) * S.get(3, 1, 1, 1) +
  //                           R.get(1, 1, 2, 0) * S.get(0, 2, 1, 1) +
  //                           R.get(1, 1, 2, 1) * S.get(1, 2, 1, 1) +
  //                           R.get(1, 1, 2, 2) * S.get(2, 2, 1, 1) +
  //                           R.get(1, 1, 2, 3) * S.get(3, 2, 1, 1) +
  //                           R.get(1, 1, 3, 0) * S.get(0, 3, 1, 1) +
  //                           R.get(1, 1, 3, 1) * S.get(1, 3, 1, 1) +
  //                           R.get(1, 1, 3, 2) * S.get(2, 3, 1, 1) +
  //                           R.get(1, 1, 3, 3) * S.get(3, 3, 1, 1) +
  //                           R.get(1, 2, 0, 0) * S.get(0, 0, 2, 1) +
  //                           R.get(1, 2, 0, 1) * S.get(1, 0, 2, 1) +
  //                           R.get(1, 2, 0, 2) * S.get(2, 0, 2, 1) +
  //                           R.get(1, 2, 0, 3) * S.get(3, 0, 2, 1) +
  //                           R.get(1, 2, 1, 0) * S.get(0, 1, 2, 1) +
  //                           R.get(1, 2, 1, 1) * S.get(1, 1, 2, 1) +
  //                           R.get(1, 2, 1, 2) * S.get(2, 1, 2, 1) +
  //                           R.get(1, 2, 1, 3) * S.get(3, 1, 2, 1) +
  //                           R.get(1, 2, 2, 0) * S.get(0, 2, 2, 1) +
  //                           R.get(1, 2, 2, 1) * S.get(1, 2, 2, 1) +
  //                           R.get(1, 2, 2, 2) * S.get(2, 2, 2, 1) +
  //                           R.get(1, 2, 2, 3) * S.get(3, 2, 2, 1) +
  //                           R.get(1, 2, 3, 0) * S.get(0, 3, 2, 1) +
  //                           R.get(1, 2, 3, 1) * S.get(1, 3, 2, 1) +
  //                           R.get(1, 2, 3, 2) * S.get(2, 3, 2, 1) +
  //                           R.get(1, 2, 3, 3) * S.get(3, 3, 2, 1) +
  //                           R.get(1, 3, 0, 0) * S.get(0, 0, 3, 1) +
  //                           R.get(1, 3, 0, 1) * S.get(1, 0, 3, 1) +
  //                           R.get(1, 3, 0, 2) * S.get(2, 0, 3, 1) +
  //                           R.get(1, 3, 0, 3) * S.get(3, 0, 3, 1) +
  //                           R.get(1, 3, 1, 0) * S.get(0, 1, 3, 1) +
  //                           R.get(1, 3, 1, 1) * S.get(1, 1, 3, 1) +
  //                           R.get(1, 3, 1, 2) * S.get(2, 1, 3, 1) +
  //                           R.get(1, 3, 1, 3) * S.get(3, 1, 3, 1) +
  //                           R.get(1, 3, 2, 0) * S.get(0, 2, 3, 1) +
  //                           R.get(1, 3, 2, 1) * S.get(1, 2, 3, 1) +
  //                           R.get(1, 3, 2, 2) * S.get(2, 2, 3, 1) +
  //                           R.get(1, 3, 2, 3) * S.get(3, 2, 3, 1) +
  //                           R.get(1, 3, 3, 0) * S.get(0, 3, 3, 1) +
  //                           R.get(1, 3, 3, 1) * S.get(1, 3, 3, 1) +
  //                           R.get(1, 3, 3, 2) * S.get(2, 3, 3, 1) +
  //                           R.get(1, 3, 3, 3) * S.get(3, 3, 3, 1) +
  //                           R.get(2, 0, 0, 0) * S.get(0, 0, 0, 2) +
  //                           R.get(2, 0, 0, 1) * S.get(1, 0, 0, 2) +
  //                           R.get(2, 0, 0, 2) * S.get(2, 0, 0, 2) +
  //                           R.get(2, 0, 0, 3) * S.get(3, 0, 0, 2) +
  //                           R.get(2, 0, 1, 0) * S.get(0, 1, 0, 2) +
  //                           R.get(2, 0, 1, 1) * S.get(1, 1, 0, 2) +
  //                           R.get(2, 0, 1, 2) * S.get(2, 1, 0, 2) +
  //                           R.get(2, 0, 1, 3) * S.get(3, 1, 0, 2) +
  //                           R.get(2, 0, 2, 0) * S.get(0, 2, 0, 2) +
  //                           R.get(2, 0, 2, 1) * S.get(1, 2, 0, 2) +
  //                           R.get(2, 0, 2, 2) * S.get(2, 2, 0, 2) +
  //                           R.get(2, 0, 2, 3) * S.get(3, 2, 0, 2) +
  //                           R.get(2, 0, 3, 0) * S.get(0, 3, 0, 2) +
  //                           R.get(2, 0, 3, 1) * S.get(1, 3, 0, 2) +
  //                           R.get(2, 0, 3, 2) * S.get(2, 3, 0, 2) +
  //                           R.get(2, 0, 3, 3) * S.get(3, 3, 0, 2) +
  //                           R.get(2, 1, 0, 0) * S.get(0, 0, 1, 2) +
  //                           R.get(2, 1, 0, 1) * S.get(1, 0, 1, 2) +
  //                           R.get(2, 1, 0, 2) * S.get(2, 0, 1, 2) +
  //                           R.get(2, 1, 0, 3) * S.get(3, 0, 1, 2) +
  //                           R.get(2, 1, 1, 0) * S.get(0, 1, 1, 2) +
  //                           R.get(2, 1, 1, 1) * S.get(1, 1, 1, 2) +
  //                           R.get(2, 1, 1, 2) * S.get(2, 1, 1, 2) +
  //                           R.get(2, 1, 1, 3) * S.get(3, 1, 1, 2) +
  //                           R.get(2, 1, 2, 0) * S.get(0, 2, 1, 2) +
  //                           R.get(2, 1, 2, 1) * S.get(1, 2, 1, 2) +
  //                           R.get(2, 1, 2, 2) * S.get(2, 2, 1, 2) +
  //                           R.get(2, 1, 2, 3) * S.get(3, 2, 1, 2) +
  //                           R.get(2, 1, 3, 0) * S.get(0, 3, 1, 2) +
  //                           R.get(2, 1, 3, 1) * S.get(1, 3, 1, 2) +
  //                           R.get(2, 1, 3, 2) * S.get(2, 3, 1, 2) +
  //                           R.get(2, 1, 3, 3) * S.get(3, 3, 1, 2) +
  //                           R.get(2, 2, 0, 0) * S.get(0, 0, 2, 2) +
  //                           R.get(2, 2, 0, 1) * S.get(1, 0, 2, 2) +
  //                           R.get(2, 2, 0, 2) * S.get(2, 0, 2, 2) +
  //                           R.get(2, 2, 0, 3) * S.get(3, 0, 2, 2) +
  //                           R.get(2, 2, 1, 0) * S.get(0, 1, 2, 2) +
  //                           R.get(2, 2, 1, 1) * S.get(1, 1, 2, 2) +
  //                           R.get(2, 2, 1, 2) * S.get(2, 1, 2, 2) +
  //                           R.get(2, 2, 1, 3) * S.get(3, 1, 2, 2) +
  //                           R.get(2, 2, 2, 0) * S.get(0, 2, 2, 2) +
  //                           R.get(2, 2, 2, 1) * S.get(1, 2, 2, 2) +
  //                           R.get(2, 2, 2, 2) * S.get(2, 2, 2, 2) +
  //                           R.get(2, 2, 2, 3) * S.get(3, 2, 2, 2) +
  //                           R.get(2, 2, 3, 0) * S.get(0, 3, 2, 2) +
  //                           R.get(2, 2, 3, 1) * S.get(1, 3, 2, 2) +
  //                           R.get(2, 2, 3, 2) * S.get(2, 3, 2, 2) +
  //                           R.get(2, 2, 3, 3) * S.get(3, 3, 2, 2) +
  //                           R.get(2, 3, 0, 0) * S.get(0, 0, 3, 2) +
  //                           R.get(2, 3, 0, 1) * S.get(1, 0, 3, 2) +
  //                           R.get(2, 3, 0, 2) * S.get(2, 0, 3, 2) +
  //                           R.get(2, 3, 0, 3) * S.get(3, 0, 3, 2) +
  //                           R.get(2, 3, 1, 0) * S.get(0, 1, 3, 2) +
  //                           R.get(2, 3, 1, 1) * S.get(1, 1, 3, 2) +
  //                           R.get(2, 3, 1, 2) * S.get(2, 1, 3, 2) +
  //                           R.get(2, 3, 1, 3) * S.get(3, 1, 3, 2) +
  //                           R.get(2, 3, 2, 0) * S.get(0, 2, 3, 2) +
  //                           R.get(2, 3, 2, 1) * S.get(1, 2, 3, 2) +
  //                           R.get(2, 3, 2, 2) * S.get(2, 2, 3, 2) +
  //                           R.get(2, 3, 2, 3) * S.get(3, 2, 3, 2) +
  //                           R.get(2, 3, 3, 0) * S.get(0, 3, 3, 2) +
  //                           R.get(2, 3, 3, 1) * S.get(1, 3, 3, 2) +
  //                           R.get(2, 3, 3, 2) * S.get(2, 3, 3, 2) +
  //                           R.get(2, 3, 3, 3) * S.get(3, 3, 3, 2) +
  //                           R.get(3, 0, 0, 0) * S.get(0, 0, 0, 3) +
  //                           R.get(3, 0, 0, 1) * S.get(1, 0, 0, 3) +
  //                           R.get(3, 0, 0, 2) * S.get(2, 0, 0, 3) +
  //                           R.get(3, 0, 0, 3) * S.get(3, 0, 0, 3) +
  //                           R.get(3, 0, 1, 0) * S.get(0, 1, 0, 3) +
  //                           R.get(3, 0, 1, 1) * S.get(1, 1, 0, 3) +
  //                           R.get(3, 0, 1, 2) * S.get(2, 1, 0, 3) +
  //                           R.get(3, 0, 1, 3) * S.get(3, 1, 0, 3) +
  //                           R.get(3, 0, 2, 0) * S.get(0, 2, 0, 3) +
  //                           R.get(3, 0, 2, 1) * S.get(1, 2, 0, 3) +
  //                           R.get(3, 0, 2, 2) * S.get(2, 2, 0, 3) +
  //                           R.get(3, 0, 2, 3) * S.get(3, 2, 0, 3) +
  //                           R.get(3, 0, 3, 0) * S.get(0, 3, 0, 3) +
  //                           R.get(3, 0, 3, 1) * S.get(1, 3, 0, 3) +
  //                           R.get(3, 0, 3, 2) * S.get(2, 3, 0, 3) +
  //                           R.get(3, 0, 3, 3) * S.get(3, 3, 0, 3) +
  //                           R.get(3, 1, 0, 0) * S.get(0, 0, 1, 3) +
  //                           R.get(3, 1, 0, 1) * S.get(1, 0, 1, 3) +
  //                           R.get(3, 1, 0, 2) * S.get(2, 0, 1, 3) +
  //                           R.get(3, 1, 0, 3) * S.get(3, 0, 1, 3) +
  //                           R.get(3, 1, 1, 0) * S.get(0, 1, 1, 3) +
  //                           R.get(3, 1, 1, 1) * S.get(1, 1, 1, 3) +
  //                           R.get(3, 1, 1, 2) * S.get(2, 1, 1, 3) +
  //                           R.get(3, 1, 1, 3) * S.get(3, 1, 1, 3) +
  //                           R.get(3, 1, 2, 0) * S.get(0, 2, 1, 3) +
  //                           R.get(3, 1, 2, 1) * S.get(1, 2, 1, 3) +
  //                           R.get(3, 1, 2, 2) * S.get(2, 2, 1, 3) +
  //                           R.get(3, 1, 2, 3) * S.get(3, 2, 1, 3) +
  //                           R.get(3, 1, 3, 0) * S.get(0, 3, 1, 3) +
  //                           R.get(3, 1, 3, 1) * S.get(1, 3, 1, 3) +
  //                           R.get(3, 1, 3, 2) * S.get(2, 3, 1, 3) +
  //                           R.get(3, 1, 3, 3) * S.get(3, 3, 1, 3) +
  //                           R.get(3, 2, 0, 0) * S.get(0, 0, 2, 3) +
  //                           R.get(3, 2, 0, 1) * S.get(1, 0, 2, 3) +
  //                           R.get(3, 2, 0, 2) * S.get(2, 0, 2, 3) +
  //                           R.get(3, 2, 0, 3) * S.get(3, 0, 2, 3) +
  //                           R.get(3, 2, 1, 0) * S.get(0, 1, 2, 3) +
  //                           R.get(3, 2, 1, 1) * S.get(1, 1, 2, 3) +
  //                           R.get(3, 2, 1, 2) * S.get(2, 1, 2, 3) +
  //                           R.get(3, 2, 1, 3) * S.get(3, 1, 2, 3) +
  //                           R.get(3, 2, 2, 0) * S.get(0, 2, 2, 3) +
  //                           R.get(3, 2, 2, 1) * S.get(1, 2, 2, 3) +
  //                           R.get(3, 2, 2, 2) * S.get(2, 2, 2, 3) +
  //                           R.get(3, 2, 2, 3) * S.get(3, 2, 2, 3) +
  //                           R.get(3, 2, 3, 0) * S.get(0, 3, 2, 3) +
  //                           R.get(3, 2, 3, 1) * S.get(1, 3, 2, 3) +
  //                           R.get(3, 2, 3, 2) * S.get(2, 3, 2, 3) +
  //                           R.get(3, 2, 3, 3) * S.get(3, 3, 2, 3) +
  //                           R.get(3, 3, 0, 0) * S.get(0, 0, 3, 3) +
  //                           R.get(3, 3, 0, 1) * S.get(1, 0, 3, 3) +
  //                           R.get(3, 3, 0, 2) * S.get(2, 0, 3, 3) +
  //                           R.get(3, 3, 0, 3) * S.get(3, 0, 3, 3) +
  //                           R.get(3, 3, 1, 0) * S.get(0, 1, 3, 3) +
  //                           R.get(3, 3, 1, 1) * S.get(1, 1, 3, 3) +
  //                           R.get(3, 3, 1, 2) * S.get(2, 1, 3, 3) +
  //                           R.get(3, 3, 1, 3) * S.get(3, 1, 3, 3) +
  //                           R.get(3, 3, 2, 0) * S.get(0, 2, 3, 3) +
  //                           R.get(3, 3, 2, 1) * S.get(1, 2, 3, 3) +
  //                           R.get(3, 3, 2, 2) * S.get(2, 2, 3, 3) +
  //                           R.get(3, 3, 2, 3) * S.get(3, 2, 3, 3) +
  //                           R.get(3, 3, 3, 0) * S.get(0, 3, 3, 3) +
  //                           R.get(3, 3, 3, 1) * S.get(1, 3, 3, 3) +
  //                           R.get(3, 3, 3, 2) * S.get(2, 3, 3, 3) +
  //                           R.get(3, 3, 3, 3) * S.get(3, 3, 3, 3);

  // CHECK_ITERABLE_APPROX(get(L), result);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DataVector", "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);
  test_large_datavector_expression(
      make_not_null(&generator),
      DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
