// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <climits>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename Generator, typename DataType>
void test_levi_civita(const gsl::not_null<Generator*> generator,
                      const DataType& used_for_size) {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  const auto R =
      make_with_random_values<tnsr::ii<DataType, 3, Frame::Inertial>>(
          generator, make_not_null(&distribution), used_for_size);

  // \f$L_{il}{}^j{}_k = \epsilon_i{}^j R_{kl}\f$
  const Tensor<DataType, Symmetry<3, 1, 2, 1>,
               index_list<SpatialIndex<2, UpLo::Lo, Frame::LeviCivitaSymbol>,
                          SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<2, UpLo::Up, Frame::LeviCivitaSymbol>,
                          SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      e_R = TensorExpressions::evaluate<ti_i, ti_l, ti_J, ti_k>(
          te_levi_civita_2d(ti_i, ti_J) * R(ti_k, ti_l));

  auto e_iJ = make_with_value<tnsr::iJ<DataType, 2, Frame::LeviCivitaSymbol>>(
      used_for_size, 0.0);
  get<0, 1>(e_iJ) = 1.0;
  get<1, 0>(e_iJ) = -1.0;

  for (size_t i = 0; i < 2; i++) {
    for (size_t l = 0; l < 3; l++) {
      for (size_t j = 0; j < 2; j++) {
        for (size_t k = 0; k < 3; k++) {
          CHECK_ITERABLE_APPROX(e_R.get(i, l, j, k),
                                e_iJ.get(i, j) * R.get(k, l));
        }
      }
    }
  }

  const auto S = make_with_random_values<tnsr::A<DataType, 3, Frame::Inertial>>(
      generator, make_not_null(&distribution), used_for_size);

  // \f$L^{il}{}_{kj} = S^i \epsilon_{ijk}\f$
  const Tensor<DataType, Symmetry<4, 3, 2, 1>,
               index_list<SpatialIndex<3, UpLo::Up, Frame::LeviCivitaSymbol>,
                          SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                          SpatialIndex<3, UpLo::Lo, Frame::LeviCivitaSymbol>,
                          SpatialIndex<3, UpLo::Lo, Frame::LeviCivitaSymbol>>>
      S_e = TensorExpressions::evaluate<ti_I, ti_L, ti_k, ti_j>(
          S(ti_L) * te_levi_civita_3d(ti_j, ti_k, ti_I));

  auto e_ijk = make_with_value<tnsr::ijk<DataType, 3, Frame::LeviCivitaSymbol>>(
      used_for_size, 0.0);
  get<0, 1, 2>(e_ijk) = 1.0;
  get<2, 0, 1>(e_ijk) = 1.0;
  get<1, 2, 0>(e_ijk) = 1.0;
  get<2, 1, 0>(e_ijk) = -1.0;
  get<0, 2, 1>(e_ijk) = -1.0;
  get<1, 0, 2>(e_ijk) = -1.0;

  for (size_t l = 0; l < 3; l++) {
    for (size_t i = 0; i < 3; i++) {
      for (size_t k = 0; k < 3; k++) {
        for (size_t j = 0; j < 3; j++) {
          CHECK_ITERABLE_APPROX(S_e.get(i, l, k, j),
                                S.get(l + 1) * e_ijk.get(j, k, i));
        }
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.LeviCivita",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  test_levi_civita(make_not_null(&generator),
                   std::numeric_limits<double>::signaling_NaN());
  test_levi_civita(make_not_null(&generator),
                   DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
