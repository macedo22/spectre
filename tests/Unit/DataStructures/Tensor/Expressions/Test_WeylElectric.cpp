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
#include "DataStructures/VectorImpl.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> weyl_electric(
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_ricci,
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept {
  tnsr::ii<DataType, SpatialDim, Frame> weyl_electric_part{};
  weyl_electric<SpatialDim, Frame, DataType>(make_not_null(&weyl_electric_part),
                                             spatial_ricci, extrinsic_curvature,
                                             inverse_spatial_metric);
  return weyl_electric_part;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void weyl_electric(
    const gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*>
        weyl_electric_part,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_ricci,
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept {
  *weyl_electric_part = spatial_ricci;
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      for (size_t k = 0; k < SpatialDim; ++k) {
        for (size_t l = 0; l < SpatialDim; ++l) {
          weyl_electric_part->get(i, j) +=
              inverse_spatial_metric.get(k, l) *
              (extrinsic_curvature.get(k, l) * extrinsic_curvature.get(i, j) -
               extrinsic_curvature.get(i, l) * extrinsic_curvature.get(k, j));
        }
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.TEWeylElectric",
                  "[DataStructures][Unit]") {
  Tensor<double, Symmetry<1, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      spatial_ricci{};
  std::iota(spatial_ricci.begin(), spatial_ricci.end(), 0.0);

  Tensor<double, Symmetry<1, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      extrinsic_curvature{};
  std::iota(extrinsic_curvature.begin(), extrinsic_curvature.end(), 0.0);

  Tensor<double, Symmetry<1, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Up, Frame::Inertial>>>
      inverse_spatial_metric{};
  std::iota(inverse_spatial_metric.begin(), inverse_spatial_metric.end(), 0.0);

  auto og_weyl_electric_tensor = weyl_electric<3, Frame::Inertial, double>(
      spatial_ricci, extrinsic_curvature, inverse_spatial_metric);

  Tensor<double, Symmetry<2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      TE_weyl_electric_tensor = TensorExpressions::evaluate<ti_i_t, ti_j_t>(
          spatial_ricci(ti_i, ti_j) +
          (extrinsic_curvature(ti_k, ti_l) *
            inverse_spatial_metric(ti_K, ti_L) *
           extrinsic_curvature(ti_i, ti_j)) -
          (extrinsic_curvature(ti_i, ti_l) *
            inverse_spatial_metric(ti_K, ti_L) *
           extrinsic_curvature(ti_k, ti_j)));

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      CHECK(og_weyl_electric_tensor.get(i, j) ==
            TE_weyl_electric_tensor.get(i, j));
    }
  }
}
