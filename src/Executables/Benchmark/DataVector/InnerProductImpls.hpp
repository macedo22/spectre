// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Executables/Benchmark/BenchmarkHelpers.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"

// Implementations benchmarked
struct BenchmarkImpl {
  // tensor types in tensor equation being benchmarked
  using DataType = DataVector;
  static constexpr size_t Dim = 3;
  using result_type = Scalar<DataType>;
  using R_type =
      Tensor<DataType, Symmetry<4, 3, 2, 1>,
             index_list<SpacetimeIndex<Dim, UpLo::Up, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Up, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Up, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Up, Frame::Inertial>>>;
  using S_type =
      Tensor<DataType, Symmetry<4, 3, 2, 1>,
             index_list<SpacetimeIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Lo, Frame::Inertial>>>;

  static inline std::array<std::array<size_t, 4>, 256> multi_indices = {
      {{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, 2}, {0, 0, 0, 3}, {0, 0, 1, 0},
       {0, 0, 1, 1}, {0, 0, 1, 2}, {0, 0, 1, 3}, {0, 0, 2, 0}, {0, 0, 2, 1},
       {0, 0, 2, 2}, {0, 0, 2, 3}, {0, 0, 3, 0}, {0, 0, 3, 1}, {0, 0, 3, 2},
       {0, 0, 3, 3}, {0, 1, 0, 0}, {0, 1, 0, 1}, {0, 1, 0, 2}, {0, 1, 0, 3},
       {0, 1, 1, 0}, {0, 1, 1, 1}, {0, 1, 1, 2}, {0, 1, 1, 3}, {0, 1, 2, 0},
       {0, 1, 2, 1}, {0, 1, 2, 2}, {0, 1, 2, 3}, {0, 1, 3, 0}, {0, 1, 3, 1},
       {0, 1, 3, 2}, {0, 1, 3, 3}, {0, 2, 0, 0}, {0, 2, 0, 1}, {0, 2, 0, 2},
       {0, 2, 0, 3}, {0, 2, 1, 0}, {0, 2, 1, 1}, {0, 2, 1, 2}, {0, 2, 1, 3},
       {0, 2, 2, 0}, {0, 2, 2, 1}, {0, 2, 2, 2}, {0, 2, 2, 3}, {0, 2, 3, 0},
       {0, 2, 3, 1}, {0, 2, 3, 2}, {0, 2, 3, 3}, {0, 3, 0, 0}, {0, 3, 0, 1},
       {0, 3, 0, 2}, {0, 3, 0, 3}, {0, 3, 1, 0}, {0, 3, 1, 1}, {0, 3, 1, 2},
       {0, 3, 1, 3}, {0, 3, 2, 0}, {0, 3, 2, 1}, {0, 3, 2, 2}, {0, 3, 2, 3},
       {0, 3, 3, 0}, {0, 3, 3, 1}, {0, 3, 3, 2}, {0, 3, 3, 3}, {1, 0, 0, 0},
       {1, 0, 0, 1}, {1, 0, 0, 2}, {1, 0, 0, 3}, {1, 0, 1, 0}, {1, 0, 1, 1},
       {1, 0, 1, 2}, {1, 0, 1, 3}, {1, 0, 2, 0}, {1, 0, 2, 1}, {1, 0, 2, 2},
       {1, 0, 2, 3}, {1, 0, 3, 0}, {1, 0, 3, 1}, {1, 0, 3, 2}, {1, 0, 3, 3},
       {1, 1, 0, 0}, {1, 1, 0, 1}, {1, 1, 0, 2}, {1, 1, 0, 3}, {1, 1, 1, 0},
       {1, 1, 1, 1}, {1, 1, 1, 2}, {1, 1, 1, 3}, {1, 1, 2, 0}, {1, 1, 2, 1},
       {1, 1, 2, 2}, {1, 1, 2, 3}, {1, 1, 3, 0}, {1, 1, 3, 1}, {1, 1, 3, 2},
       {1, 1, 3, 3}, {1, 2, 0, 0}, {1, 2, 0, 1}, {1, 2, 0, 2}, {1, 2, 0, 3},
       {1, 2, 1, 0}, {1, 2, 1, 1}, {1, 2, 1, 2}, {1, 2, 1, 3}, {1, 2, 2, 0},
       {1, 2, 2, 1}, {1, 2, 2, 2}, {1, 2, 2, 3}, {1, 2, 3, 0}, {1, 2, 3, 1},
       {1, 2, 3, 2}, {1, 2, 3, 3}, {1, 3, 0, 0}, {1, 3, 0, 1}, {1, 3, 0, 2},
       {1, 3, 0, 3}, {1, 3, 1, 0}, {1, 3, 1, 1}, {1, 3, 1, 2}, {1, 3, 1, 3},
       {1, 3, 2, 0}, {1, 3, 2, 1}, {1, 3, 2, 2}, {1, 3, 2, 3}, {1, 3, 3, 0},
       {1, 3, 3, 1}, {1, 3, 3, 2}, {1, 3, 3, 3}, {2, 0, 0, 0}, {2, 0, 0, 1},
       {2, 0, 0, 2}, {2, 0, 0, 3}, {2, 0, 1, 0}, {2, 0, 1, 1}, {2, 0, 1, 2},
       {2, 0, 1, 3}, {2, 0, 2, 0}, {2, 0, 2, 1}, {2, 0, 2, 2}, {2, 0, 2, 3},
       {2, 0, 3, 0}, {2, 0, 3, 1}, {2, 0, 3, 2}, {2, 0, 3, 3}, {2, 1, 0, 0},
       {2, 1, 0, 1}, {2, 1, 0, 2}, {2, 1, 0, 3}, {2, 1, 1, 0}, {2, 1, 1, 1},
       {2, 1, 1, 2}, {2, 1, 1, 3}, {2, 1, 2, 0}, {2, 1, 2, 1}, {2, 1, 2, 2},
       {2, 1, 2, 3}, {2, 1, 3, 0}, {2, 1, 3, 1}, {2, 1, 3, 2}, {2, 1, 3, 3},
       {2, 2, 0, 0}, {2, 2, 0, 1}, {2, 2, 0, 2}, {2, 2, 0, 3}, {2, 2, 1, 0},
       {2, 2, 1, 1}, {2, 2, 1, 2}, {2, 2, 1, 3}, {2, 2, 2, 0}, {2, 2, 2, 1},
       {2, 2, 2, 2}, {2, 2, 2, 3}, {2, 2, 3, 0}, {2, 2, 3, 1}, {2, 2, 3, 2},
       {2, 2, 3, 3}, {2, 3, 0, 0}, {2, 3, 0, 1}, {2, 3, 0, 2}, {2, 3, 0, 3},
       {2, 3, 1, 0}, {2, 3, 1, 1}, {2, 3, 1, 2}, {2, 3, 1, 3}, {2, 3, 2, 0},
       {2, 3, 2, 1}, {2, 3, 2, 2}, {2, 3, 2, 3}, {2, 3, 3, 0}, {2, 3, 3, 1},
       {2, 3, 3, 2}, {2, 3, 3, 3}, {3, 0, 0, 0}, {3, 0, 0, 1}, {3, 0, 0, 2},
       {3, 0, 0, 3}, {3, 0, 1, 0}, {3, 0, 1, 1}, {3, 0, 1, 2}, {3, 0, 1, 3},
       {3, 0, 2, 0}, {3, 0, 2, 1}, {3, 0, 2, 2}, {3, 0, 2, 3}, {3, 0, 3, 0},
       {3, 0, 3, 1}, {3, 0, 3, 2}, {3, 0, 3, 3}, {3, 1, 0, 0}, {3, 1, 0, 1},
       {3, 1, 0, 2}, {3, 1, 0, 3}, {3, 1, 1, 0}, {3, 1, 1, 1}, {3, 1, 1, 2},
       {3, 1, 1, 3}, {3, 1, 2, 0}, {3, 1, 2, 1}, {3, 1, 2, 2}, {3, 1, 2, 3},
       {3, 1, 3, 0}, {3, 1, 3, 1}, {3, 1, 3, 2}, {3, 1, 3, 3}, {3, 2, 0, 0},
       {3, 2, 0, 1}, {3, 2, 0, 2}, {3, 2, 0, 3}, {3, 2, 1, 0}, {3, 2, 1, 1},
       {3, 2, 1, 2}, {3, 2, 1, 3}, {3, 2, 2, 0}, {3, 2, 2, 1}, {3, 2, 2, 2},
       {3, 2, 2, 3}, {3, 2, 3, 0}, {3, 2, 3, 1}, {3, 2, 3, 2}, {3, 2, 3, 3},
       {3, 3, 0, 0}, {3, 3, 0, 1}, {3, 3, 0, 2}, {3, 3, 0, 3}, {3, 3, 1, 0},
       {3, 3, 1, 1}, {3, 3, 1, 2}, {3, 3, 1, 3}, {3, 3, 2, 0}, {3, 3, 2, 1},
       {3, 3, 2, 2}, {3, 3, 2, 3}, {3, 3, 3, 0}, {3, 3, 3, 1}, {3, 3, 3, 2},
       {3, 3, 3, 3}}};

  SPECTRE_ALWAYS_INLINE static void loop_impl(
      gsl::not_null<result_type*> result, const R_type& R, const S_type& S) {
    destructive_resize_components(result, get_size(get<0, 0, 0, 0>(R)));
    get(*result) = 0.0;

    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        for (size_t c = 0; c < Dim + 1; c++) {
          for (size_t d = 0; d < Dim + 1; d++) {
            get(*result) += R.get(a, b, c, d) * S.get(a, b, c, d);
          }
        }
      }
    }
  }

  SPECTRE_ALWAYS_INLINE static void oneline_impl(
      gsl::not_null<result_type*> result, const R_type& R, const S_type& S) {
    destructive_resize_components(result, get_size(get<0, 0, 0, 0>(R)));
    get(*result) = R.get(0, 0, 0, 0) * S.get(0, 0, 0, 0) +
                   R.get(0, 0, 0, 1) * S.get(0, 0, 0, 1) +
                   R.get(0, 0, 0, 2) * S.get(0, 0, 0, 2) +
                   R.get(0, 0, 0, 3) * S.get(0, 0, 0, 3) +
                   R.get(0, 0, 1, 0) * S.get(0, 0, 1, 0) +
                   R.get(0, 0, 1, 1) * S.get(0, 0, 1, 1) +
                   R.get(0, 0, 1, 2) * S.get(0, 0, 1, 2) +
                   R.get(0, 0, 1, 3) * S.get(0, 0, 1, 3) +
                   R.get(0, 0, 2, 0) * S.get(0, 0, 2, 0) +
                   R.get(0, 0, 2, 1) * S.get(0, 0, 2, 1) +
                   R.get(0, 0, 2, 2) * S.get(0, 0, 2, 2) +
                   R.get(0, 0, 2, 3) * S.get(0, 0, 2, 3) +
                   R.get(0, 0, 3, 0) * S.get(0, 0, 3, 0) +
                   R.get(0, 0, 3, 1) * S.get(0, 0, 3, 1) +
                   R.get(0, 0, 3, 2) * S.get(0, 0, 3, 2) +
                   R.get(0, 0, 3, 3) * S.get(0, 0, 3, 3) +
                   R.get(0, 1, 0, 0) * S.get(0, 1, 0, 0) +
                   R.get(0, 1, 0, 1) * S.get(0, 1, 0, 1) +
                   R.get(0, 1, 0, 2) * S.get(0, 1, 0, 2) +
                   R.get(0, 1, 0, 3) * S.get(0, 1, 0, 3) +
                   R.get(0, 1, 1, 0) * S.get(0, 1, 1, 0) +
                   R.get(0, 1, 1, 1) * S.get(0, 1, 1, 1) +
                   R.get(0, 1, 1, 2) * S.get(0, 1, 1, 2) +
                   R.get(0, 1, 1, 3) * S.get(0, 1, 1, 3) +
                   R.get(0, 1, 2, 0) * S.get(0, 1, 2, 0) +
                   R.get(0, 1, 2, 1) * S.get(0, 1, 2, 1) +
                   R.get(0, 1, 2, 2) * S.get(0, 1, 2, 2) +
                   R.get(0, 1, 2, 3) * S.get(0, 1, 2, 3) +
                   R.get(0, 1, 3, 0) * S.get(0, 1, 3, 0) +
                   R.get(0, 1, 3, 1) * S.get(0, 1, 3, 1) +
                   R.get(0, 1, 3, 2) * S.get(0, 1, 3, 2) +
                   R.get(0, 1, 3, 3) * S.get(0, 1, 3, 3) +
                   R.get(0, 2, 0, 0) * S.get(0, 2, 0, 0) +
                   R.get(0, 2, 0, 1) * S.get(0, 2, 0, 1) +
                   R.get(0, 2, 0, 2) * S.get(0, 2, 0, 2) +
                   R.get(0, 2, 0, 3) * S.get(0, 2, 0, 3) +
                   R.get(0, 2, 1, 0) * S.get(0, 2, 1, 0) +
                   R.get(0, 2, 1, 1) * S.get(0, 2, 1, 1) +
                   R.get(0, 2, 1, 2) * S.get(0, 2, 1, 2) +
                   R.get(0, 2, 1, 3) * S.get(0, 2, 1, 3) +
                   R.get(0, 2, 2, 0) * S.get(0, 2, 2, 0) +
                   R.get(0, 2, 2, 1) * S.get(0, 2, 2, 1) +
                   R.get(0, 2, 2, 2) * S.get(0, 2, 2, 2) +
                   R.get(0, 2, 2, 3) * S.get(0, 2, 2, 3) +
                   R.get(0, 2, 3, 0) * S.get(0, 2, 3, 0) +
                   R.get(0, 2, 3, 1) * S.get(0, 2, 3, 1) +
                   R.get(0, 2, 3, 2) * S.get(0, 2, 3, 2) +
                   R.get(0, 2, 3, 3) * S.get(0, 2, 3, 3) +
                   R.get(0, 3, 0, 0) * S.get(0, 3, 0, 0) +
                   R.get(0, 3, 0, 1) * S.get(0, 3, 0, 1) +
                   R.get(0, 3, 0, 2) * S.get(0, 3, 0, 2) +
                   R.get(0, 3, 0, 3) * S.get(0, 3, 0, 3) +
                   R.get(0, 3, 1, 0) * S.get(0, 3, 1, 0) +
                   R.get(0, 3, 1, 1) * S.get(0, 3, 1, 1) +
                   R.get(0, 3, 1, 2) * S.get(0, 3, 1, 2) +
                   R.get(0, 3, 1, 3) * S.get(0, 3, 1, 3) +
                   R.get(0, 3, 2, 0) * S.get(0, 3, 2, 0) +
                   R.get(0, 3, 2, 1) * S.get(0, 3, 2, 1) +
                   R.get(0, 3, 2, 2) * S.get(0, 3, 2, 2) +
                   R.get(0, 3, 2, 3) * S.get(0, 3, 2, 3) +
                   R.get(0, 3, 3, 0) * S.get(0, 3, 3, 0) +
                   R.get(0, 3, 3, 1) * S.get(0, 3, 3, 1) +
                   R.get(0, 3, 3, 2) * S.get(0, 3, 3, 2) +
                   R.get(0, 3, 3, 3) * S.get(0, 3, 3, 3) +
                   R.get(1, 0, 0, 0) * S.get(1, 0, 0, 0) +
                   R.get(1, 0, 0, 1) * S.get(1, 0, 0, 1) +
                   R.get(1, 0, 0, 2) * S.get(1, 0, 0, 2) +
                   R.get(1, 0, 0, 3) * S.get(1, 0, 0, 3) +
                   R.get(1, 0, 1, 0) * S.get(1, 0, 1, 0) +
                   R.get(1, 0, 1, 1) * S.get(1, 0, 1, 1) +
                   R.get(1, 0, 1, 2) * S.get(1, 0, 1, 2) +
                   R.get(1, 0, 1, 3) * S.get(1, 0, 1, 3) +
                   R.get(1, 0, 2, 0) * S.get(1, 0, 2, 0) +
                   R.get(1, 0, 2, 1) * S.get(1, 0, 2, 1) +
                   R.get(1, 0, 2, 2) * S.get(1, 0, 2, 2) +
                   R.get(1, 0, 2, 3) * S.get(1, 0, 2, 3) +
                   R.get(1, 0, 3, 0) * S.get(1, 0, 3, 0) +
                   R.get(1, 0, 3, 1) * S.get(1, 0, 3, 1) +
                   R.get(1, 0, 3, 2) * S.get(1, 0, 3, 2) +
                   R.get(1, 0, 3, 3) * S.get(1, 0, 3, 3) +
                   R.get(1, 1, 0, 0) * S.get(1, 1, 0, 0) +
                   R.get(1, 1, 0, 1) * S.get(1, 1, 0, 1) +
                   R.get(1, 1, 0, 2) * S.get(1, 1, 0, 2) +
                   R.get(1, 1, 0, 3) * S.get(1, 1, 0, 3) +
                   R.get(1, 1, 1, 0) * S.get(1, 1, 1, 0) +
                   R.get(1, 1, 1, 1) * S.get(1, 1, 1, 1) +
                   R.get(1, 1, 1, 2) * S.get(1, 1, 1, 2) +
                   R.get(1, 1, 1, 3) * S.get(1, 1, 1, 3) +
                   R.get(1, 1, 2, 0) * S.get(1, 1, 2, 0) +
                   R.get(1, 1, 2, 1) * S.get(1, 1, 2, 1) +
                   R.get(1, 1, 2, 2) * S.get(1, 1, 2, 2) +
                   R.get(1, 1, 2, 3) * S.get(1, 1, 2, 3) +
                   R.get(1, 1, 3, 0) * S.get(1, 1, 3, 0) +
                   R.get(1, 1, 3, 1) * S.get(1, 1, 3, 1) +
                   R.get(1, 1, 3, 2) * S.get(1, 1, 3, 2) +
                   R.get(1, 1, 3, 3) * S.get(1, 1, 3, 3) +
                   R.get(1, 2, 0, 0) * S.get(1, 2, 0, 0) +
                   R.get(1, 2, 0, 1) * S.get(1, 2, 0, 1) +
                   R.get(1, 2, 0, 2) * S.get(1, 2, 0, 2) +
                   R.get(1, 2, 0, 3) * S.get(1, 2, 0, 3) +
                   R.get(1, 2, 1, 0) * S.get(1, 2, 1, 0) +
                   R.get(1, 2, 1, 1) * S.get(1, 2, 1, 1) +
                   R.get(1, 2, 1, 2) * S.get(1, 2, 1, 2) +
                   R.get(1, 2, 1, 3) * S.get(1, 2, 1, 3) +
                   R.get(1, 2, 2, 0) * S.get(1, 2, 2, 0) +
                   R.get(1, 2, 2, 1) * S.get(1, 2, 2, 1) +
                   R.get(1, 2, 2, 2) * S.get(1, 2, 2, 2) +
                   R.get(1, 2, 2, 3) * S.get(1, 2, 2, 3) +
                   R.get(1, 2, 3, 0) * S.get(1, 2, 3, 0) +
                   R.get(1, 2, 3, 1) * S.get(1, 2, 3, 1) +
                   R.get(1, 2, 3, 2) * S.get(1, 2, 3, 2) +
                   R.get(1, 2, 3, 3) * S.get(1, 2, 3, 3) +
                   R.get(1, 3, 0, 0) * S.get(1, 3, 0, 0) +
                   R.get(1, 3, 0, 1) * S.get(1, 3, 0, 1) +
                   R.get(1, 3, 0, 2) * S.get(1, 3, 0, 2) +
                   R.get(1, 3, 0, 3) * S.get(1, 3, 0, 3) +
                   R.get(1, 3, 1, 0) * S.get(1, 3, 1, 0) +
                   R.get(1, 3, 1, 1) * S.get(1, 3, 1, 1) +
                   R.get(1, 3, 1, 2) * S.get(1, 3, 1, 2) +
                   R.get(1, 3, 1, 3) * S.get(1, 3, 1, 3) +
                   R.get(1, 3, 2, 0) * S.get(1, 3, 2, 0) +
                   R.get(1, 3, 2, 1) * S.get(1, 3, 2, 1) +
                   R.get(1, 3, 2, 2) * S.get(1, 3, 2, 2) +
                   R.get(1, 3, 2, 3) * S.get(1, 3, 2, 3) +
                   R.get(1, 3, 3, 0) * S.get(1, 3, 3, 0) +
                   R.get(1, 3, 3, 1) * S.get(1, 3, 3, 1) +
                   R.get(1, 3, 3, 2) * S.get(1, 3, 3, 2) +
                   R.get(1, 3, 3, 3) * S.get(1, 3, 3, 3) +
                   R.get(2, 0, 0, 0) * S.get(2, 0, 0, 0) +
                   R.get(2, 0, 0, 1) * S.get(2, 0, 0, 1) +
                   R.get(2, 0, 0, 2) * S.get(2, 0, 0, 2) +
                   R.get(2, 0, 0, 3) * S.get(2, 0, 0, 3) +
                   R.get(2, 0, 1, 0) * S.get(2, 0, 1, 0) +
                   R.get(2, 0, 1, 1) * S.get(2, 0, 1, 1) +
                   R.get(2, 0, 1, 2) * S.get(2, 0, 1, 2) +
                   R.get(2, 0, 1, 3) * S.get(2, 0, 1, 3) +
                   R.get(2, 0, 2, 0) * S.get(2, 0, 2, 0) +
                   R.get(2, 0, 2, 1) * S.get(2, 0, 2, 1) +
                   R.get(2, 0, 2, 2) * S.get(2, 0, 2, 2) +
                   R.get(2, 0, 2, 3) * S.get(2, 0, 2, 3) +
                   R.get(2, 0, 3, 0) * S.get(2, 0, 3, 0) +
                   R.get(2, 0, 3, 1) * S.get(2, 0, 3, 1) +
                   R.get(2, 0, 3, 2) * S.get(2, 0, 3, 2) +
                   R.get(2, 0, 3, 3) * S.get(2, 0, 3, 3) +
                   R.get(2, 1, 0, 0) * S.get(2, 1, 0, 0) +
                   R.get(2, 1, 0, 1) * S.get(2, 1, 0, 1) +
                   R.get(2, 1, 0, 2) * S.get(2, 1, 0, 2) +
                   R.get(2, 1, 0, 3) * S.get(2, 1, 0, 3) +
                   R.get(2, 1, 1, 0) * S.get(2, 1, 1, 0) +
                   R.get(2, 1, 1, 1) * S.get(2, 1, 1, 1) +
                   R.get(2, 1, 1, 2) * S.get(2, 1, 1, 2) +
                   R.get(2, 1, 1, 3) * S.get(2, 1, 1, 3) +
                   R.get(2, 1, 2, 0) * S.get(2, 1, 2, 0) +
                   R.get(2, 1, 2, 1) * S.get(2, 1, 2, 1) +
                   R.get(2, 1, 2, 2) * S.get(2, 1, 2, 2) +
                   R.get(2, 1, 2, 3) * S.get(2, 1, 2, 3) +
                   R.get(2, 1, 3, 0) * S.get(2, 1, 3, 0) +
                   R.get(2, 1, 3, 1) * S.get(2, 1, 3, 1) +
                   R.get(2, 1, 3, 2) * S.get(2, 1, 3, 2) +
                   R.get(2, 1, 3, 3) * S.get(2, 1, 3, 3) +
                   R.get(2, 2, 0, 0) * S.get(2, 2, 0, 0) +
                   R.get(2, 2, 0, 1) * S.get(2, 2, 0, 1) +
                   R.get(2, 2, 0, 2) * S.get(2, 2, 0, 2) +
                   R.get(2, 2, 0, 3) * S.get(2, 2, 0, 3) +
                   R.get(2, 2, 1, 0) * S.get(2, 2, 1, 0) +
                   R.get(2, 2, 1, 1) * S.get(2, 2, 1, 1) +
                   R.get(2, 2, 1, 2) * S.get(2, 2, 1, 2) +
                   R.get(2, 2, 1, 3) * S.get(2, 2, 1, 3) +
                   R.get(2, 2, 2, 0) * S.get(2, 2, 2, 0) +
                   R.get(2, 2, 2, 1) * S.get(2, 2, 2, 1) +
                   R.get(2, 2, 2, 2) * S.get(2, 2, 2, 2) +
                   R.get(2, 2, 2, 3) * S.get(2, 2, 2, 3) +
                   R.get(2, 2, 3, 0) * S.get(2, 2, 3, 0) +
                   R.get(2, 2, 3, 1) * S.get(2, 2, 3, 1) +
                   R.get(2, 2, 3, 2) * S.get(2, 2, 3, 2) +
                   R.get(2, 2, 3, 3) * S.get(2, 2, 3, 3) +
                   R.get(2, 3, 0, 0) * S.get(2, 3, 0, 0) +
                   R.get(2, 3, 0, 1) * S.get(2, 3, 0, 1) +
                   R.get(2, 3, 0, 2) * S.get(2, 3, 0, 2) +
                   R.get(2, 3, 0, 3) * S.get(2, 3, 0, 3) +
                   R.get(2, 3, 1, 0) * S.get(2, 3, 1, 0) +
                   R.get(2, 3, 1, 1) * S.get(2, 3, 1, 1) +
                   R.get(2, 3, 1, 2) * S.get(2, 3, 1, 2) +
                   R.get(2, 3, 1, 3) * S.get(2, 3, 1, 3) +
                   R.get(2, 3, 2, 0) * S.get(2, 3, 2, 0) +
                   R.get(2, 3, 2, 1) * S.get(2, 3, 2, 1) +
                   R.get(2, 3, 2, 2) * S.get(2, 3, 2, 2) +
                   R.get(2, 3, 2, 3) * S.get(2, 3, 2, 3) +
                   R.get(2, 3, 3, 0) * S.get(2, 3, 3, 0) +
                   R.get(2, 3, 3, 1) * S.get(2, 3, 3, 1) +
                   R.get(2, 3, 3, 2) * S.get(2, 3, 3, 2) +
                   R.get(2, 3, 3, 3) * S.get(2, 3, 3, 3) +
                   R.get(3, 0, 0, 0) * S.get(3, 0, 0, 0) +
                   R.get(3, 0, 0, 1) * S.get(3, 0, 0, 1) +
                   R.get(3, 0, 0, 2) * S.get(3, 0, 0, 2) +
                   R.get(3, 0, 0, 3) * S.get(3, 0, 0, 3) +
                   R.get(3, 0, 1, 0) * S.get(3, 0, 1, 0) +
                   R.get(3, 0, 1, 1) * S.get(3, 0, 1, 1) +
                   R.get(3, 0, 1, 2) * S.get(3, 0, 1, 2) +
                   R.get(3, 0, 1, 3) * S.get(3, 0, 1, 3) +
                   R.get(3, 0, 2, 0) * S.get(3, 0, 2, 0) +
                   R.get(3, 0, 2, 1) * S.get(3, 0, 2, 1) +
                   R.get(3, 0, 2, 2) * S.get(3, 0, 2, 2) +
                   R.get(3, 0, 2, 3) * S.get(3, 0, 2, 3) +
                   R.get(3, 0, 3, 0) * S.get(3, 0, 3, 0) +
                   R.get(3, 0, 3, 1) * S.get(3, 0, 3, 1) +
                   R.get(3, 0, 3, 2) * S.get(3, 0, 3, 2) +
                   R.get(3, 0, 3, 3) * S.get(3, 0, 3, 3) +
                   R.get(3, 1, 0, 0) * S.get(3, 1, 0, 0) +
                   R.get(3, 1, 0, 1) * S.get(3, 1, 0, 1) +
                   R.get(3, 1, 0, 2) * S.get(3, 1, 0, 2) +
                   R.get(3, 1, 0, 3) * S.get(3, 1, 0, 3) +
                   R.get(3, 1, 1, 0) * S.get(3, 1, 1, 0) +
                   R.get(3, 1, 1, 1) * S.get(3, 1, 1, 1) +
                   R.get(3, 1, 1, 2) * S.get(3, 1, 1, 2) +
                   R.get(3, 1, 1, 3) * S.get(3, 1, 1, 3) +
                   R.get(3, 1, 2, 0) * S.get(3, 1, 2, 0) +
                   R.get(3, 1, 2, 1) * S.get(3, 1, 2, 1) +
                   R.get(3, 1, 2, 2) * S.get(3, 1, 2, 2) +
                   R.get(3, 1, 2, 3) * S.get(3, 1, 2, 3) +
                   R.get(3, 1, 3, 0) * S.get(3, 1, 3, 0) +
                   R.get(3, 1, 3, 1) * S.get(3, 1, 3, 1) +
                   R.get(3, 1, 3, 2) * S.get(3, 1, 3, 2) +
                   R.get(3, 1, 3, 3) * S.get(3, 1, 3, 3) +
                   R.get(3, 2, 0, 0) * S.get(3, 2, 0, 0) +
                   R.get(3, 2, 0, 1) * S.get(3, 2, 0, 1) +
                   R.get(3, 2, 0, 2) * S.get(3, 2, 0, 2) +
                   R.get(3, 2, 0, 3) * S.get(3, 2, 0, 3) +
                   R.get(3, 2, 1, 0) * S.get(3, 2, 1, 0) +
                   R.get(3, 2, 1, 1) * S.get(3, 2, 1, 1) +
                   R.get(3, 2, 1, 2) * S.get(3, 2, 1, 2) +
                   R.get(3, 2, 1, 3) * S.get(3, 2, 1, 3) +
                   R.get(3, 2, 2, 0) * S.get(3, 2, 2, 0) +
                   R.get(3, 2, 2, 1) * S.get(3, 2, 2, 1) +
                   R.get(3, 2, 2, 2) * S.get(3, 2, 2, 2) +
                   R.get(3, 2, 2, 3) * S.get(3, 2, 2, 3) +
                   R.get(3, 2, 3, 0) * S.get(3, 2, 3, 0) +
                   R.get(3, 2, 3, 1) * S.get(3, 2, 3, 1) +
                   R.get(3, 2, 3, 2) * S.get(3, 2, 3, 2) +
                   R.get(3, 2, 3, 3) * S.get(3, 2, 3, 3) +
                   R.get(3, 3, 0, 0) * S.get(3, 3, 0, 0) +
                   R.get(3, 3, 0, 1) * S.get(3, 3, 0, 1) +
                   R.get(3, 3, 0, 2) * S.get(3, 3, 0, 2) +
                   R.get(3, 3, 0, 3) * S.get(3, 3, 0, 3) +
                   R.get(3, 3, 1, 0) * S.get(3, 3, 1, 0) +
                   R.get(3, 3, 1, 1) * S.get(3, 3, 1, 1) +
                   R.get(3, 3, 1, 2) * S.get(3, 3, 1, 2) +
                   R.get(3, 3, 1, 3) * S.get(3, 3, 1, 3) +
                   R.get(3, 3, 2, 0) * S.get(3, 3, 2, 0) +
                   R.get(3, 3, 2, 1) * S.get(3, 3, 2, 1) +
                   R.get(3, 3, 2, 2) * S.get(3, 3, 2, 2) +
                   R.get(3, 3, 2, 3) * S.get(3, 3, 2, 3) +
                   R.get(3, 3, 3, 0) * S.get(3, 3, 3, 0) +
                   R.get(3, 3, 3, 1) * S.get(3, 3, 3, 1) +
                   R.get(3, 3, 3, 2) * S.get(3, 3, 3, 2) +
                   R.get(3, 3, 3, 3) * S.get(3, 3, 3, 3);
  }

  template <size_t Iteration>
  SPECTRE_ALWAYS_INLINE static decltype(auto) recursive_impl_impl(
      const R_type& R, const S_type& S) {
    if constexpr (Iteration != 0) {
      return recursive_impl_impl<Iteration - 1>(R, S) +
             R.get(multi_indices[Iteration]) * S.get(multi_indices[Iteration]);
    } else {
      return R.get(multi_indices[Iteration]) * S.get(multi_indices[Iteration]);
    }
  }

  SPECTRE_ALWAYS_INLINE static void recursive_impl(
      gsl::not_null<result_type*> result, const R_type& R, const S_type& S) {
    destructive_resize_components(result, get_size(get<0, 0, 0, 0>(R)));

    get(*result) = recursive_impl_impl<255>(R, S);
  }

  template <size_t Iteration>
  SPECTRE_ALWAYS_INLINE static void recursive_plus_equals_impl_impl(
      gsl::not_null<result_type*> result, const R_type& R, const S_type& S) {
    if constexpr (Iteration != 0) {
      recursive_plus_equals_impl_impl<Iteration - 1>(result, R, S);
    }

    get(*result) +=
        R.get(multi_indices[Iteration]) * S.get(multi_indices[Iteration]);
  }

  SPECTRE_ALWAYS_INLINE static void recursive_plus_equals_impl(
      gsl::not_null<result_type*> result, const R_type& R, const S_type& S) {
    destructive_resize_components(result, get_size(get<0, 0, 0, 0>(R)));
    get(*result) = 0.0;

    recursive_plus_equals_impl_impl<255>(result, R, S);
  }
};
