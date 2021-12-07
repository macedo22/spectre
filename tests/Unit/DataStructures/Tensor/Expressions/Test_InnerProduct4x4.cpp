// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iostream>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <typename T>
struct td;

constexpr size_t Dim = 1;
using FrameType = Frame::Inertial;
template <typename DataType>
using R_type = Tensor<DataType, Symmetry<4, 3, 2, 1>,
                      index_list<SpacetimeIndex<Dim, UpLo::Up, FrameType>,
                                 SpacetimeIndex<Dim, UpLo::Up, FrameType>,
                                 SpacetimeIndex<Dim, UpLo::Up, FrameType>,
                                 SpacetimeIndex<Dim, UpLo::Up, FrameType>>>;
template <typename DataType>
using S_type = Tensor<DataType, Symmetry<4, 3, 2, 1>,
                      index_list<SpacetimeIndex<Dim, UpLo::Lo, FrameType>,
                                 SpacetimeIndex<Dim, UpLo::Lo, FrameType>,
                                 SpacetimeIndex<Dim, UpLo::Lo, FrameType>,
                                 SpacetimeIndex<Dim, UpLo::Lo, FrameType>>>;

template <typename DataType>
void test_te(gsl::not_null<Scalar<DataType>*> L, const R_type<DataType>& R,
             const S_type<DataType>& S) {
  TensorExpressions::evaluate(
      L, R(ti_A, ti_B, ti_C, ti_D) * S(ti_a, ti_b, ti_c, ti_d));

  // 15
  std::cout << "TEDouble::add_ops_ : " << TEDouble::add_ops_ << std::endl;
  // 0
  std::cout << "TEDouble::add_equals_ops_ : " << TEDouble::add_equals_ops_
            << std::endl;
  // 16
  std::cout << "TEDouble::mult_ops_ : " << TEDouble::mult_ops_ << std::endl;
}

template <typename DataType>
void test_loop(gsl::not_null<Scalar<DataType>*> L, const R_type<DataType>& R,
               const S_type<DataType>& S) {
  get(*L) = 0.0;
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      for (size_t c = 0; c < Dim + 1; c++) {
        for (size_t d = 0; d < Dim + 1; d++) {
          get(*L) += R.get(a, b, c, d) * S.get(a, b, c, d);
        }
      }
    }
  }

  // 0
  std::cout << "LoopDouble::add_ops_ : " << LoopDouble::add_ops_ << std::endl;
  // 16
  std::cout << "LoopDouble::add_equals_ops_ : " << LoopDouble::add_equals_ops_
            << std::endl;
  // 16
  std::cout << "LoopDouble::mult_ops_ : " << LoopDouble::mult_ops_ << std::endl;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.InnerProduct4x4",
                  "[Unit][DataStructures]") {
  R_type<TEDouble> TE_R{};
  S_type<TEDouble> TE_S{};
  Scalar<TEDouble> TE_L{};
  test_te(make_not_null(&TE_L), TE_R, TE_S);

  R_type<LoopDouble> Loop_R{};
  S_type<LoopDouble> Loop_S{};
  Scalar<LoopDouble> Loop_L{};
  test_loop(make_not_null(&Loop_L), Loop_R, Loop_S);
}
