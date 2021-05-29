// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylMagnetic.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_include <boost/preprocessor/arithmetic/dec.hpp>
// IWYU pragma: no_include <boost/preprocessor/repetition/enum.hpp>
// IWYU pragma: no_include <boost/preprocessor/tuple/reverse.hpp>

namespace {
template <size_t SpatialDim, typename DataType>
void test_compute_item_in_databox(const DataType& used_for_size) noexcept {
  TestHelpers::db::test_compute_tag<
      gr::Tags::WeylMagneticCompute<SpatialDim, Frame::Inertial, DataType>>(
      "WeylMagnetic");

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-3.0, 3.0);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);

  const auto grad_extrinsic_curvature =
      make_with_random_values<tnsr::ijj<DataType, SpatialDim>>(
          nn_generator, nn_distribution, used_for_size);
  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<SpatialDim>(nn_generator,
                                                         used_for_size);

  const auto box = db::create<
      db::AddSimpleTags<
          ::Tags::deriv<gr::Tags::ExtrinsicCurvature<SpatialDim,
                                                     Frame::Inertial, DataType>,
                        tmpl::size_t<SpatialDim>, Frame::Inertial>,
          gr::Tags::SpatialMetric<SpatialDim, Frame::Inertial, DataType>>,
      db::AddComputeTags<gr::Tags::WeylMagneticCompute<
          SpatialDim, Frame::Inertial, DataType>>>(grad_extrinsic_curvature,
                                                   spatial_metric);

  const auto expected =
      gr::weyl_magnetic(grad_extrinsic_curvature, spatial_metric);
  CHECK_ITERABLE_APPROX(
      (db::get<gr::Tags::WeylMagnetic<SpatialDim, Frame::Inertial, DataType>>(
          box)),
      expected);
}
template <size_t SpatialDim, typename DataType>
void test_weyl_magnetic(const DataType& used_for_size) {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-3.0, 3.0);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);

  const auto grad_extrinsic_curvature =
      make_with_random_values<tnsr::ijj<DataType, SpatialDim>>(
          nn_generator, nn_distribution, used_for_size);
  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<SpatialDim>(nn_generator,
                                                         used_for_size);

  CHECK_ITERABLE_APPROX(
      (gr::weyl_magnetic<SpatialDim, Frame::Inertial, DataType>(
          grad_extrinsic_curvature, spatial_metric)),
      (pypp::call<tnsr::ii<DataType, SpatialDim, Frame::Inertial>>(
          "WeylMagnetic", "weyl_magnetic_tensor", grad_extrinsic_curvature,
          spatial_metric)));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.WeylMagnetic",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;

  test_compute_item_in_databox<3>(d);
  test_compute_item_in_databox<3>(dv);
  test_weyl_magnetic<3>(d);
  test_weyl_magnetic<3>(dv);
}
