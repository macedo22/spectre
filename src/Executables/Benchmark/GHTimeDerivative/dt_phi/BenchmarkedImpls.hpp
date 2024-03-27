// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/Product.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Executables/Benchmark/BenchmarkHelpers.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

// Implementations benchmarked
template <typename DataType, size_t Dim>
struct BenchmarkImpl {
  // tensor types in tensor equation being benchmarked
  using dt_phi_type = tnsr::iaa<DataType, Dim>;
  using pi_type = tnsr::aa<DataType, Dim>;
  using d_pi_type = tnsr::iaa<DataType, Dim>;
  using phi_two_normals_type = tnsr::i<DataType, Dim>;
  using three_index_constraint_type = tnsr::iaa<DataType, Dim>;
  using gamma2_type = Scalar<DataType>;
  using phi_one_normal_type = tnsr::ia<DataType, Dim>;
  using phi_1_up_type = tnsr::Iaa<DataType, Dim>;
  using lapse_type = Scalar<DataType>;
  using shift_type = tnsr::I<DataType, Dim>;
  using d_phi_type = tnsr::ijaa<DataType, Dim>;

  // manual implementation benchmarked that takes LHS tensor as arg
  SPECTRE_ALWAYS_INLINE static void manual_impl_lhs_arg(
      gsl::not_null<dt_phi_type*> dt_phi, const pi_type& pi,
      const d_pi_type& d_pi, const phi_two_normals_type& phi_two_normals,
      const three_index_constraint_type& three_index_constraint,
      const gamma2_type& gamma2, const phi_one_normal_type& phi_one_normal,
      const phi_1_up_type& phi_1_up, const lapse_type& lapse,
      const shift_type& shift, const d_phi_type& d_phi) {
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t mu = 0; mu < Dim + 1; ++mu) {
        for (size_t nu = mu; nu < Dim + 1; ++nu) {
          dt_phi->get(i, mu, nu) =
              0.5 * pi.get(mu, nu) * phi_two_normals.get(i) -
              d_pi.get(i, mu, nu) +
              get(gamma2) * three_index_constraint.get(i, mu, nu);
          for (size_t n = 0; n < Dim; ++n) {
            dt_phi->get(i, mu, nu) +=
                phi_one_normal.get(i, n + 1) * phi_1_up.get(n, mu, nu);
          }

          dt_phi->get(i, mu, nu) *= get(lapse);
          for (size_t m = 0; m < Dim; ++m) {
            dt_phi->get(i, mu, nu) += shift.get(m) * d_phi.get(m, i, mu, nu);
          }
        }
      }
    }
  }

  // TensorExpression implementation benchmarked that takes LHS tensor as arg
  template <size_t CaseNumber>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg(
      gsl::not_null<dt_phi_type*> dt_phi, const pi_type& pi,
      const d_pi_type& d_pi, const phi_two_normals_type& phi_two_normals,
      const three_index_constraint_type& three_index_constraint,
      const gamma2_type& gamma2, const phi_one_normal_type& phi_one_normal,
      const phi_1_up_type& phi_1_up, const lapse_type& lapse,
      const shift_type& shift, const d_phi_type& d_phi);

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<1>(
      gsl::not_null<dt_phi_type*> dt_phi, const pi_type& pi,
      const d_pi_type& d_pi, const phi_two_normals_type& phi_two_normals,
      const three_index_constraint_type& three_index_constraint,
      const gamma2_type& gamma2, const phi_one_normal_type& phi_one_normal,
      const phi_1_up_type& phi_1_up, const lapse_type& lapse,
      const shift_type& shift, const d_phi_type& d_phi) {
    tenex::evaluate<ti::i, ti::a, ti::b>(
        dt_phi,
        (0.5 * pi(ti::a, ti::b) * phi_two_normals(ti::i) - d_pi(ti::i, ti::a, ti::b) +
         gamma2() * three_index_constraint(ti::i, ti::a, ti::b) +
         phi_one_normal(ti::i, ti::j) * phi_1_up(ti::J, ti::a, ti::b)) *
                lapse() +
            shift(ti::K) * d_phi(ti::k, ti::i, ti::a, ti::b));
  }
};
