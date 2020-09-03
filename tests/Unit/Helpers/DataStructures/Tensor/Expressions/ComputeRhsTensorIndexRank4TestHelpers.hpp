// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once


#include <cstddef>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TMPL.hpp"

// Check each element of each mapping
template <typename Datatype, typename RhsSymmetry,
          typename RhsTensorIndexTypeList, typename TensorIndexA,
          typename TensorIndexB, typename TensorIndexC, typename TensorIndexD>
void test_compute_rhs_tensor_index_rank_4(const TensorIndexA& tensorindex_a,
                                          const TensorIndexB& tensorindex_b,
                                          const TensorIndexC& tensorindex_c,
                                          const TensorIndexD& tensorindex_d) {
  const Tensor<Datatype, RhsSymmetry, RhsTensorIndexTypeList> rhs_tensor{};
  const auto R_abcd =
      rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d);

  const size_t dim_a = tmpl::at_c<RhsTensorIndexTypeList, 0>::dim;
  const size_t dim_b = tmpl::at_c<RhsTensorIndexTypeList, 1>::dim;
  const size_t dim_c = tmpl::at_c<RhsTensorIndexTypeList, 2>::dim;
  const size_t dim_d = tmpl::at_c<RhsTensorIndexTypeList, 3>::dim;

  const std::array<size_t, 4> index_order_abcd = {
      TensorIndexA::value, TensorIndexB::value, TensorIndexC::value,
      TensorIndexD::value};
  const std::array<size_t, 4> index_order_abdc = {
      TensorIndexA::value, TensorIndexB::value, TensorIndexD::value,
      TensorIndexC::value};
  const std::array<size_t, 4> index_order_acbd = {
      TensorIndexA::value, TensorIndexC::value, TensorIndexB::value,
      TensorIndexD::value};
  const std::array<size_t, 4> index_order_acdb = {
      TensorIndexA::value, TensorIndexC::value, TensorIndexD::value,
      TensorIndexB::value};
  const std::array<size_t, 4> index_order_adbc = {
      TensorIndexA::value, TensorIndexD::value, TensorIndexB::value,
      TensorIndexC::value};
  const std::array<size_t, 4> index_order_adcb = {
      TensorIndexA::value, TensorIndexD::value, TensorIndexC::value,
      TensorIndexB::value};

  const std::array<size_t, 4> index_order_bacd = {
      TensorIndexB::value, TensorIndexA::value, TensorIndexC::value,
      TensorIndexD::value};
  const std::array<size_t, 4> index_order_badc = {
      TensorIndexB::value, TensorIndexA::value, TensorIndexD::value,
      TensorIndexC::value};
  const std::array<size_t, 4> index_order_bcad = {
      TensorIndexB::value, TensorIndexC::value, TensorIndexA::value,
      TensorIndexD::value};
  const std::array<size_t, 4> index_order_bcda = {
      TensorIndexB::value, TensorIndexC::value, TensorIndexD::value,
      TensorIndexA::value};
  const std::array<size_t, 4> index_order_bdac = {
      TensorIndexB::value, TensorIndexD::value, TensorIndexA::value,
      TensorIndexC::value};
  const std::array<size_t, 4> index_order_bdca = {
      TensorIndexB::value, TensorIndexD::value, TensorIndexC::value,
      TensorIndexA::value};

  const std::array<size_t, 4> index_order_cabd = {
      TensorIndexC::value, TensorIndexA::value, TensorIndexB::value,
      TensorIndexD::value};
  const std::array<size_t, 4> index_order_cadb = {
      TensorIndexC::value, TensorIndexA::value, TensorIndexD::value,
      TensorIndexB::value};
  const std::array<size_t, 4> index_order_cbad = {
      TensorIndexC::value, TensorIndexB::value, TensorIndexA::value,
      TensorIndexD::value};
  const std::array<size_t, 4> index_order_cbda = {
      TensorIndexC::value, TensorIndexB::value, TensorIndexD::value,
      TensorIndexA::value};
  const std::array<size_t, 4> index_order_cdab = {
      TensorIndexC::value, TensorIndexD::value, TensorIndexA::value,
      TensorIndexB::value};
  const std::array<size_t, 4> index_order_cdba = {
      TensorIndexC::value, TensorIndexD::value, TensorIndexB::value,
      TensorIndexA::value};

  const std::array<size_t, 4> index_order_dabc = {
      TensorIndexD::value, TensorIndexA::value, TensorIndexB::value,
      TensorIndexC::value};
  const std::array<size_t, 4> index_order_dacb = {
      TensorIndexD::value, TensorIndexA::value, TensorIndexC::value,
      TensorIndexB::value};
  const std::array<size_t, 4> index_order_dbac = {
      TensorIndexD::value, TensorIndexB::value, TensorIndexA::value,
      TensorIndexC::value};
  const std::array<size_t, 4> index_order_dbca = {
      TensorIndexD::value, TensorIndexB::value, TensorIndexC::value,
      TensorIndexA::value};
  const std::array<size_t, 4> index_order_dcab = {
      TensorIndexD::value, TensorIndexC::value, TensorIndexA::value,
      TensorIndexB::value};
  const std::array<size_t, 4> index_order_dcba = {
      TensorIndexD::value, TensorIndexC::value, TensorIndexB::value,
      TensorIndexA::value};

  for (size_t i = 0; i < dim_a; i++) {
    for (size_t j = 0; j < dim_b; j++) {
      for (size_t k = 0; k < dim_c; k++) {
        for (size_t l = 0; l < dim_d; l++) {
          const std::array<size_t, 4> ijkl = {i, j, k, l};
          const std::array<size_t, 4> ijlk = {i, j, l, k};
          const std::array<size_t, 4> ikjl = {i, k, j, l};
          const std::array<size_t, 4> iklj = {i, k, l, j};
          const std::array<size_t, 4> iljk = {i, l, j, k};
          const std::array<size_t, 4> ilkj = {i, l, k, j};

          const std::array<size_t, 4> jikl = {j, i, k, l};
          const std::array<size_t, 4> jilk = {j, i, l, k};
          const std::array<size_t, 4> jkil = {j, k, i, l};
          const std::array<size_t, 4> jkli = {j, k, l, i};
          const std::array<size_t, 4> jlik = {j, l, i, k};
          const std::array<size_t, 4> jlki = {j, l, k, i};

          const std::array<size_t, 4> kijl = {k, i, j, l};
          const std::array<size_t, 4> kilj = {k, i, l, j};
          const std::array<size_t, 4> kjil = {k, j, i, l};
          const std::array<size_t, 4> kjli = {k, j, l, i};
          const std::array<size_t, 4> klij = {k, l, i, j};
          const std::array<size_t, 4> klji = {k, l, j, i};

          const std::array<size_t, 4> lijk = {l, i, j, k};
          const std::array<size_t, 4> likj = {l, i, k, j};
          const std::array<size_t, 4> ljik = {l, j, i, k};
          const std::array<size_t, 4> ljki = {l, j, k, i};
          const std::array<size_t, 4> lkij = {l, k, i, j};
          const std::array<size_t, 4> lkji = {l, k, j, i};

          // RHS = {a, b, c, d}
          CHECK(R_abcd.template compute_rhs_tensor_index<4>(
                    index_order_abcd, index_order_abcd, ijkl) == ijkl);
          CHECK(R_abcd.template compute_rhs_tensor_index<4>(
                    index_order_abdc, index_order_abcd, ijkl) == ijlk);
          CHECK(R_abcd.template compute_rhs_tensor_index<4>(
                    index_order_acbd, index_order_abcd, ijkl) == ikjl);
          CHECK(R_abcd.template compute_rhs_tensor_index<4>(
                    index_order_acdb, index_order_abcd, ijkl) == iljk);
          CHECK(R_abcd.template compute_rhs_tensor_index<4>(
                    index_order_adbc, index_order_abcd, ijkl) == iklj);
          CHECK(R_abcd.template compute_rhs_tensor_index<4>(
                    index_order_adcb, index_order_abcd, ijkl) == ilkj);

          CHECK(R_abcd.template compute_rhs_tensor_index<4>(
                    index_order_bacd, index_order_abcd, ijkl) == jikl);
          CHECK(R_abcd.template compute_rhs_tensor_index<4>(
                    index_order_badc, index_order_abcd, ijkl) == jilk);
          CHECK(R_abcd.template compute_rhs_tensor_index<4>(
                    index_order_bcad, index_order_abcd, ijkl) == kijl);
          CHECK(R_abcd.template compute_rhs_tensor_index<4>(
                    index_order_bcda, index_order_abcd, ijkl) == lijk);
          CHECK(R_abcd.template compute_rhs_tensor_index<4>(
                    index_order_bdac, index_order_abcd, ijkl) == kilj);
          CHECK(R_abcd.template compute_rhs_tensor_index<4>(
                    index_order_bdca, index_order_abcd, ijkl) == likj);

          CHECK(R_abcd.template compute_rhs_tensor_index<4>(
                    index_order_cabd, index_order_abcd, ijkl) == jkil);
          CHECK(R_abcd.template compute_rhs_tensor_index<4>(
                    index_order_cadb, index_order_abcd, ijkl) == jlik);
          CHECK(R_abcd.template compute_rhs_tensor_index<4>(
                    index_order_cbad, index_order_abcd, ijkl) == kjil);
          CHECK(R_abcd.template compute_rhs_tensor_index<4>(
                    index_order_cbda, index_order_abcd, ijkl) == ljik);
          CHECK(R_abcd.template compute_rhs_tensor_index<4>(
                    index_order_cdab, index_order_abcd, ijkl) == klij);
          CHECK(R_abcd.template compute_rhs_tensor_index<4>(
                    index_order_cdba, index_order_abcd, ijkl) == lkij);

          CHECK(R_abcd.template compute_rhs_tensor_index<4>(
                    index_order_dabc, index_order_abcd, ijkl) == jkli);
          CHECK(R_abcd.template compute_rhs_tensor_index<4>(
                    index_order_dacb, index_order_abcd, ijkl) == jlki);
          CHECK(R_abcd.template compute_rhs_tensor_index<4>(
                    index_order_dbac, index_order_abcd, ijkl) == kjli);
          CHECK(R_abcd.template compute_rhs_tensor_index<4>(
                    index_order_dbca, index_order_abcd, ijkl) == ljki);
          CHECK(R_abcd.template compute_rhs_tensor_index<4>(
                    index_order_dcab, index_order_abcd, ijkl) == klji);
          CHECK(R_abcd.template compute_rhs_tensor_index<4>(
                    index_order_dcba, index_order_abcd, ijkl) == lkji);
        }
      }
    }
  }
}
