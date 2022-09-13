// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Psi4.hpp"

#include <cstddef>

#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/ProjectionOperators.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylPropagating.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace gr {
template <size_t SpatialDim, typename Frame, typename DataType>
void psi_4(
    const gsl::not_null<Scalar<DataType>*> psi_4_result,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_ricci,
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature,
    const tnsr::ijj<DataType, SpatialDim, Frame>& cov_deriv_extrinsic_curvature,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    // const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::I<DataType, SpatialDim, Frame>& inertial_coords) {
  // const tnsr::ii<DataType, SpatialDim, Frame>& weyl_electric,
  // const tnsr::ii<DataType, SpatialDim, Frame>& weyl_magnetic) {
  // let's make this section for the projection tensor
  // so we'll be using the Projection Operators that are already coded up
  // cause we don't need to reinvent the wheel here.
  // They are in ProjectionOperators.hpp
  const auto magnitude_cartesian =
      Scalar<DataType>(magnitude(inertial_coords, spatial_metric));
  const auto r_hat = tenex::evaluate<ti::j>(inertial_coords(ti::I) *
                                            spatial_metric(ti::i, ti::j) /
                                            magnitude_cartesian());
  const auto projection_tensor =
      transverse_projection_operator(spatial_metric, r_hat);
  const auto raised_r_hat = tenex::evaluate<ti::I>(
      r_hat(ti::j) * inverse_spatial_metric(ti::I, ti::J));
  const auto inverse_projection_tensor =
      transverse_projection_operator(inverse_spatial_metric, raised_r_hat);
  // documentation says a and b but here since they're all spatial, we'll do
  // a = k and b = l
  const auto projection_up_lo = tenex::evaluate<ti::K, ti::i>(
      projection_tensor(ti::i, ti::j) * inverse_spatial_metric(ti::K, ti::J));

  const auto u8_plus = gr::weyl_propagating(
      spatial_ricci, extrinsic_curvature, inverse_spatial_metric,
      cov_deriv_extrinsic_curvature, raised_r_hat, inverse_projection_tensor,
      projection_tensor, projection_up_lo, 1);
  // // weyl portion of U^8+
  // tnsr::ij<DataType, SpatialDim, Frame> mag_norm_cross_product{};
  // for (size_t i = 0, i < 3; ++i) {
  //   for (size_t j = 0, j < 3; ++j) {
  //     for (LeviCivitaIterator<3> it; it; ++it;) {
  //       mag_norm_cross_product.get(i, j) +=
  //           it.sign() * gsl::at(spatial_metric, it[0], i) *
  //           gsl::at(r_hat, it[2]) *
  //           gsl::at(weyl_magnetic, it[1], j / sqrt_det_spatial_metric);
  //     }
  //   }
  // }
  // // U^8+_ij
  // const auto u8_plus = tennex::evaluate<ti::i, ti::j>(
  //     (((0.5 * projection_up_lo(ti::K, ti::i) *
  //        projection_up_low(ti::L, ti::j)) +
  //       (0.5 * projection_up_lo(ti::L, ti::i) *
  //        projection_up_lo(ti::K, ti::j))) -
  //      (0.5 * inverse_projection_tensor(ti::K, ti::L) *
  //       projection_tensor(ti::i, ti::j))) *
  //     mag_norm_cross_product(ti::k, ti::l));

  // section for Psi4 now
  // only computing the real part of psi4 for now to test and make sure
  // everything is working.
  // making x_hat and y_hat from the mappedcoordinates.
  auto x_coord = make_with_value<tnsr::I<DataType, SpatialDim, Frame>>(
      get<0, 0>(inverse_spatial_metric), 0.0);
  x_coord.get(0) += inertial_coords.get(0);
  const auto magnitude_x = Scalar<DataType>(magnitude(x_coord, spatial_metric));
  const auto x_hat = tenex::evaluate<ti::I>(x_coord(ti::I) / magnitude_x());
  auto y_coord = make_with_value<tnsr::I<DataType, SpatialDim, Frame>>(
      get<0, 0>(inverse_spatial_metric), 0.0);
  y_coord.get(1) += inertial_coords.get(1);
  const auto magnitude_y = Scalar<DataType>(magnitude(y_coord, spatial_metric));
  const auto y_hat = tenex::evaluate<ti::I>(y_coord(ti::I) / magnitude_y());

  // making the real portion of psi4 only for now.
  tenex::evaluate(psi_4_result, 0.5 * u8_plus(ti::i, ti::j) *
                                    (y_hat(ti::I) * y_hat(ti::J) -
                                     x_hat(ti::I) * x_hat(ti::J)));
  // let's make this section for the U^8+- tensor
  //
  // inside here however, we'll have to have sections because
  // this quantity is definitely a bit more complicated than the others
  // I've done so far. For instance, I think we'll to compute the
  //(P(^a_iP^b)_j - P^abP_ij) on it's own then compute the
  //(E_ab -+ e_i^kl*n_d*B_kl)
  // also side note, we're just gonna work on coding up and returning
  // the real part of Psi4 for now and we'll worry about returning the
  // imaginary part after we code up and test the real portion.
}

template <size_t SpatialDim, typename Frame, typename DataType>
Scalar<DataType> psi_4(
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_ricci,
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature,
    const tnsr::ijj<DataType, SpatialDim, Frame>& cov_deriv_extrinsic_curvature,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    // const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::I<DataType, SpatialDim, Frame>& inertial_coords) {
  // const tnsr::ii<DataType, SpatialDim, Frame>& weyl_electric,
  // const tnsr::ii<DataType, SpatialDim, Frame>& weyl_magnetic) {
  auto psi_4_result =
      make_with_value<Scalar<DataType>>(get<0, 0>(inverse_spatial_metric), 0.0);
  psi_4(make_not_null(&psi_4_result), spatial_ricci, extrinsic_curvature,
        cov_deriv_extrinsic_curvature, spatial_metric, inverse_spatial_metric,
        inertial_coords);
  return psi_4_result;
}
}  // namespace gr

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                \
  template Scalar<DTYPE(data)> gr::psi_4(                                   \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_ricci,   \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                  \
          extrinsic_curvature,                                              \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>&                 \
          cov_deriv_extrinsic_curvature,                                    \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,  \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                  \
          inverse_spatial_metric,                                           \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& inertial_coords); \
  template void gr::psi_4(                                                  \
      const gsl::not_null<Scalar<DTYPE(data)>*> psi_4_result,               \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_ricci,   \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                  \
          extrinsic_curvature,                                              \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>&                 \
          cov_deriv_extrinsic_curvature,                                    \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,  \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                  \
          inverse_spatial_metric,                                           \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& inertial_coords);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
