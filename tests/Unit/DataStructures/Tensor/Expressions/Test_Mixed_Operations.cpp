// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"
//#include "PointwiseFunctions/GeneralRelativity/WeylElectric.hpp"

#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/Product.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// R_{ab} S_B + G_a - H_{baB} T
template <typename ResultTensor, typename R_t, typename S_t, typename G_t,
          typename H_t, typename T_t>
ResultTensor compute_expected_result(const R_t& R, const S_t& S, const G_t& G,
                                     const H_t& H, const T_t& T) noexcept {
  ResultTensor expected_result{};
  const size_t dim = tmpl::front<typename R_t::index_list>::dim;
  for (size_t a = 0; a < dim; a++) {
    double expected_Rab_SB_product = 0.0;
    double expected_HbaB_contracted_value = 0.0;
    for (size_t b = 0; b < dim; b++) {
      expected_Rab_SB_product += R.get(a, b) * S.get(b);
      expected_HbaB_contracted_value += H.get(b, a, b);
    }
    expected_result.get(a) = expected_Rab_SB_product + G.get(a) -
                             (expected_HbaB_contracted_value * T.get());
  }

  return expected_result;
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.MixedOperations",
                  "[DataStructures][Unit]") {
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      R{};
  std::iota(R.begin(), R.end(), 0.0);

  // For S_B
  Tensor<double, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      S{};
  std::iota(S.begin(), S.end(), 0.0);

  // For G_a
  Tensor<double, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      G{};
  std::iota(G.begin(), G.end(), 0.0);

  // For H_{baB}
  Tensor<double, Symmetry<3, 2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      H{};
  std::iota(H.begin(), H.end(), 0.0);

  // For T
  const Tensor<double> T{{{-2.6}}};

  using result_tensor_type =
      Tensor<double, Symmetry<1>,
             index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>;

  result_tensor_type expected_result_tensor =
      compute_expected_result<result_tensor_type>(R, S, G, H, T);
  // R_{ab} S_B + G_a - H_{baB} T
  result_tensor_type actual_result_tensor = TensorExpressions::evaluate<ti_a_t>(
      R(ti_a, ti_b) * S(ti_B) + G(ti_a) - H(ti_b, ti_a, ti_B) * T());

  for (size_t a = 0; a < 4; a++) {
    CHECK(actual_result_tensor.get(a) == expected_result_tensor.get(a));
  }
}
