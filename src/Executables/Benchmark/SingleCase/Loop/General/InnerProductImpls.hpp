// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"

// Implementations benchmarked
template <typename DataType, size_t Dim>
struct BenchmarkImpl {
  using R_type_1x1 =
      Tensor<DataType, Symmetry<1>,
             index_list<SpacetimeIndex<Dim, UpLo::Up, Frame::Inertial>>>;
  using S_type_1x1 =
      Tensor<DataType, Symmetry<1>,
             index_list<SpacetimeIndex<Dim, UpLo::Lo, Frame::Inertial>>>;
  using R_type_2x2 =
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpacetimeIndex<Dim, UpLo::Up, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Up, Frame::Inertial>>>;
  using S_type_2x2 =
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpacetimeIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Lo, Frame::Inertial>>>;
  using R_type_3x3 =
      Tensor<DataType, Symmetry<3, 2, 1>,
             index_list<SpacetimeIndex<Dim, UpLo::Up, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Up, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Up, Frame::Inertial>>>;
  using S_type_3x3 =
      Tensor<DataType, Symmetry<3, 2, 1>,
             index_list<SpacetimeIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Lo, Frame::Inertial>>>;
  using R_type_4x4 =
      Tensor<DataType, Symmetry<4, 3, 2, 1>,
             index_list<SpacetimeIndex<Dim, UpLo::Up, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Up, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Up, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Up, Frame::Inertial>>>;
  using S_type_4x4 =
      Tensor<DataType, Symmetry<4, 3, 2, 1>,
             index_list<SpacetimeIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Lo, Frame::Inertial>>>;
  using result_type = Scalar<DataType>;

  template <typename R_type, typename S_type>
  SPECTRE_ALWAYS_INLINE static void apply(gsl::not_null<result_type*> result,
                                          const R_type& R, const S_type& S) {
    destructive_resize_components(result, get_size(R[0]));
    get(*result) = 0.0;

    if constexpr (std::is_same_v<R_type, R_type_1x1>) {
      for (size_t a = 0; a < Dim + 1; a++) {
        get(*result) += R.get(a) * S.get(a);
      }
    } else if constexpr (std::is_same_v<R_type, R_type_2x2>) {
      for (size_t a = 0; a < Dim + 1; a++) {
        for (size_t b = 0; b < Dim + 1; b++) {
          get(*result) += R.get(a, b) * S.get(a, b);
        }
      }
    } else if constexpr (std::is_same_v<R_type, R_type_3x3>) {
      for (size_t a = 0; a < Dim + 1; a++) {
        for (size_t b = 0; b < Dim + 1; b++) {
          for (size_t c = 0; c < Dim + 1; c++) {
            get(*result) += R.get(a, b, c) * S.get(a, b, c);
          }
        }
      }
    } else if constexpr (std::is_same_v<R_type, R_type_4x4>) {
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
  }
};
