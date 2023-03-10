// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace PUP {
class er;
}  // namespace PUP

namespace gsl {
template <class T>
class not_null;
}  // namespace gsl

template <typename, typename, typename>
class Tensor;
/// \endcond

namespace GeneralizedHarmonic {
template <size_t Dim>
struct ComputeUpwindTempTags {
 public:
  using temporary_tags = tmpl::list<
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1,
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2,
      gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<Dim, Frame::Inertial, DataVector>,
      gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>,
      gr::Tags::DetSpatialMetric<DataVector>>;
  using argument_tags = tmpl::list<
      gr::Tags::SpacetimeMetric<Dim>,
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1,
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2>;

  static void apply(
      const gsl::not_null<Scalar<DataVector>*> temp_gamma1,
      const gsl::not_null<Scalar<DataVector>*> temp_gamma2,
      const gsl::not_null<Scalar<DataVector>*> lapse,
      const gsl::not_null<tnsr::I<DataVector, Dim>*> shift,
      const gsl::not_null<tnsr::ii<DataVector, Dim>*> spatial_metric,
      const gsl::not_null<tnsr::II<DataVector, Dim>*> inverse_spatial_metric,
      const gsl::not_null<Scalar<DataVector>*> det_spatial_metric,
      const tnsr::aa<DataVector, Dim>& spacetime_metric,
      const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2);
};
}  // namespace GeneralizedHarmonic
