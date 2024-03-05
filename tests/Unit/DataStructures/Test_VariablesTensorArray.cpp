// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/MathWrapper.hpp"
#include "Utilities/ErrorHandling/Error.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename DataType, size_t Dim, typename Frame>
struct TagA : db::SimpleTag {
  using type = tnsr::a<DataType, Dim, Frame>;
};

template <typename DataType, size_t Dim, typename Frame>
struct TagB : db::SimpleTag {
  using type = tnsr::ij<DataType, Dim, Frame>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.VariablesTensorArray",
                  "[DataStructures][Unit]") {
  constexpr size_t number_of_grid_points = 5;
  constexpr size_t Dim = 3;
  using system = typename gh::System<3>;

  // using evolution_vars_tags =
  //     tmpl::list<gr::Tags::SpacetimeMetric<DataVector, Dim>,
  //                  Tags::Pi<DataVector, Dim>, Tags::Phi<DataVector, Dim>>;
  // using evolution_variables_tag_of_tags =
  // ::Tags::Variables<evolution_vars_tags>; using partial_derivative_tags =
  // evolution_vars_tags;

  // using variables_tag = ::Tags::Variables<
  //       tmpl::list<gr::Tags::SpacetimeMetric<DataVector, Dim>,
  //                  Tags::Pi<DataVector, Dim>, Tags::Phi<DataVector, Dim>>>;
  // using variables_tag = typename system::variables_tag;
  // using gradient_variables =
  //       tmpl::list<gr::Tags::SpacetimeMetric<DataVector, Dim>,
  //                  Tags::Pi<DataVector, Dim>, Tags::Phi<DataVector, Dim>>;
  using partial_derivative_tags = typename system::gradient_variables;
  using compute_volume_time_derivative_terms =
      typename system::compute_volume_time_derivative_terms;
  using other_terms_tags = tmpl::list<TagA<DataVector, Dim, Frame::Inertial>,
                                      TagB<DataVector, Dim, Frame::Inertial>>;

  using evolved_terms =
      tmpl::list<gr::Tags::SpacetimeMetric<DataVector, Dim>,
                 gh::Tags::Pi<DataVector, Dim>, gh::Tags::Phi<DataVector, Dim>>;
  using EvolvedVars = Variables<evolved_terms>;

  using VarsTemporaries =
      Variables<typename compute_volume_time_derivative_terms::temporary_tags>;
  using VarsPartialDerivatives =
      Variables<db::wrap_tags_in<::Tags::deriv, partial_derivative_tags,
                                 tmpl::size_t<Dim>, Frame::Inertial>>;
  using VarsOther = Variables<other_terms_tags>;

  const size_t buffer_size =
      (VarsTemporaries::number_of_independent_components +
       // for inertial derivs
       VarsPartialDerivatives::number_of_independent_components +
       // for logical derivs
       EvolvedVars::number_of_independent_components * Dim +
       VarsOther::number_of_independent_components) *
      number_of_grid_points;
  auto buffer = cpp20::make_unique_for_overwrite<double[]>(buffer_size);

  VarsTemporaries temporaries{
      &buffer[0], VarsTemporaries::number_of_independent_components *
                      number_of_grid_points};
  std::array<EvolvedVars, Dim> logical_partial_derivs{};
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(logical_partial_derivs, i)
        .set_data_ref(
            &buffer[(VarsTemporaries::number_of_independent_components +
                     EvolvedVars::number_of_independent_components * i) *
                    number_of_grid_points],
            EvolvedVars::number_of_independent_components *
                number_of_grid_points);
  }
  VarsPartialDerivatives partial_derivs{
      &buffer[(VarsTemporaries::number_of_independent_components +
               EvolvedVars::number_of_independent_components * Dim) *
              number_of_grid_points],
      VarsPartialDerivatives::number_of_independent_components *
          number_of_grid_points};
  VarsOther other_terms{
      &buffer[(VarsTemporaries::number_of_independent_components +
               EvolvedVars::number_of_independent_components * Dim +
               VarsPartialDerivatives::number_of_independent_components) *
              number_of_grid_points],
      VarsOther::number_of_independent_components * number_of_grid_points};

  // assign logical deriv tensors to the buffer

  // gr::Tags::SpacetimeMetric<DataVector, Dim>
  tnsr::iaa<DataVector, Dim> logical_d_spacetime_metric{};
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = a; b < Dim + 1; b++) {
        logical_d_spacetime_metric.get(i, a, b).set_data_ref(
            &(get<gr::Tags::SpacetimeMetric<DataVector, Dim>>(
                  logical_partial_derivs[i])
                  .get(a, b)[0]),
            number_of_grid_points);
      }
    }
  }
  //   gh::Tags::Pi<DataVector, Dim>
  tnsr::iaa<DataVector, Dim> logical_d_pi{};
  for (size_t i = 0; i < Dim; i++) {
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = a; b < Dim + 1; b++) {
        logical_d_pi.get(i, a, b).set_data_ref(
            &(get<gh::Tags::Pi<DataVector, Dim>>(logical_partial_derivs[i])
                  .get(a, b)[0]),
            number_of_grid_points);
      }
    }
  }
  //   gh::Tags::Phi<DataVector, Dim>
  tnsr::ijaa<DataVector, Dim> logical_d_phi{};
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      for (size_t a = 0; a < Dim + 1; a++) {
        for (size_t b = a; b < Dim + 1; b++) {
          logical_d_phi.get(i, j, a, b)
              .set_data_ref(&(get<gh::Tags::Phi<DataVector, Dim>>(
                                  logical_partial_derivs[i])
                                  .get(j, a, b)[0]),
                            number_of_grid_points);
        }
      }
    }
  }
}
