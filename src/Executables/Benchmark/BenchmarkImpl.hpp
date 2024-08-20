// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/DuDtTempTags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Dispatch.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"

template <size_t Dim>
using gh_evolution_vars_tags =
    typename gh::System<Dim>::variables_tag::tags_list;

namespace BenchmarkImpl {
constexpr Spectral::Basis basis = Spectral::Basis::Legendre;
constexpr Spectral::Quadrature quadrature = Spectral::Quadrature::GaussLobatto;
constexpr double time = 3.4;

using DerivativeFrame = Frame::Inertial;

template <size_t Dim>
struct Kappa : db::SimpleTag {
  using type = tnsr::abb<DataVector, Dim, DerivativeFrame>;
};
template <size_t Dim>
struct Psi : db::SimpleTag {
  using type = tnsr::aa<DataVector, Dim, DerivativeFrame>;
};

template <size_t Dim>
using dt_spacetime_metric_type = tnsr::aa<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using dt_pi_type = tnsr::aa<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using dt_phi_type = tnsr::iaa<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using temp_gamma1_type = Scalar<DataVector>;
template <size_t Dim>
using temp_gamma2_type = Scalar<DataVector>;
template <size_t Dim>
using temp_gauge_function_type = tnsr::a<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using temp_spacetime_deriv_gauge_function_type =
    tnsr::ab<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using gamma1gamma2_type = Scalar<DataVector>;
template <size_t Dim>
using half_pi_two_normals_type = Scalar<DataVector>;
template <size_t Dim>
using normal_dot_gauge_constraint_type = Scalar<DataVector>;
template <size_t Dim>
using gamma1_plus_1_type = Scalar<DataVector>;
template <size_t Dim>
using pi_one_normal_type = tnsr::a<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using gauge_constraint_type = tnsr::a<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using half_phi_two_normals_type = tnsr::i<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using shift_dot_three_index_constraint_type =
    tnsr::aa<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using mesh_velocity_dot_three_index_constraint_type =
    tnsr::aa<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using phi_one_normal_type = tnsr::ia<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using pi_2_up_type = tnsr::aB<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using three_index_constraint_type = tnsr::iaa<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using phi_1_up_type = tnsr::Iaa<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using phi_3_up_type = tnsr::iaB<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using christoffel_first_kind_3_up_type =
    tnsr::abC<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using lapse_type = Scalar<DataVector>;
template <size_t Dim>
using shift_type = tnsr::I<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using inverse_spatial_metric_type = tnsr::II<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using det_spatial_metric_type = Scalar<DataVector>;
template <size_t Dim>
using sqrt_det_spatial_metric_type = Scalar<DataVector>;
template <size_t Dim>
using inverse_spacetime_metric_type =
    tnsr::AA<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using christoffel_first_kind_type = tnsr::abb<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using christoffel_second_kind_type =
    tnsr::Abb<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using trace_christoffel_type = tnsr::a<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using normal_spacetime_vector_type = tnsr::A<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using d_spacetime_metric_type = tnsr::iaa<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using d_pi_type = tnsr::iaa<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using d_phi_type = tnsr::ijaa<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using spacetime_metric_type = tnsr::aa<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using pi_type = tnsr::aa<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using phi_type = tnsr::iaa<DataVector, Dim, DerivativeFrame>;
template <size_t Dim>
using gamma0_type = Scalar<DataVector>;
template <size_t Dim>
using gamma1_type = Scalar<DataVector>;
template <size_t Dim>
using gamma2_type = Scalar<DataVector>;
template <size_t Dim>
using gauge_condition_type = gh::gauges::DampedHarmonic;
template <size_t Dim>
using inertial_coords_type = tnsr::I<DataVector, Dim, Frame::Inertial>;
template <size_t Dim>
using inverse_jacobian_type =
    InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>;
template <size_t Dim>
using mesh_velocity_type = tnsr::I<DataVector, Dim, DerivativeFrame>;

// new temporaries used in new version of time derivative
template <size_t Dim>
using logical_shift_type =
    typename gr::Tags::Shift<DataVector, Dim, Frame::ElementLogical>::type;
template <size_t Dim>
using inverse_spatial_metric_logical_1_type =
    typename gh::Tags::InverseSpatialMetricLogical1<Dim>::type;
template <size_t Dim>
using logical_mesh_velocity_type =
    typename domain::Tags::MeshVelocityWithValue<Dim,
                                                 Frame::ElementLogical>::type;
template <size_t Dim>
using shift_dot_d_spacetime_metric_type =
    typename gh::Tags::ShiftDotDSpacetimeMetric<Dim>::type;
template <size_t Dim>
using shift_dot_phi_type = typename gh::Tags::ShiftDotPhi<Dim>::type;
template <size_t Dim>
using mesh_velocity_dot_phi_type =
    typename gh::Tags::MeshVelocityDotPhi<Dim>::type;
template <size_t Dim>
using mesh_velocity_dot_d_spacetime_metric_type =
    typename gh::Tags::MeshVelocityDotDSpacetimeMetric<Dim>::type;
template <size_t Dim>
using upper_gauge_function_type = typename gh::Tags::UpperGaugeH<Dim>::type;
template <size_t Dim>
using gamma2_logical_d_spacetime_metric_minus_logical_d_pi_type =
    typename gh::Tags::Gamma2LogicalDSpacetimeMetricMinusLogicalDPi<Dim>::type;
}  // namespace BenchmarkImpl
