// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <climits>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
constexpr size_t Dim = 3;

template <typename R_type, typename S_type,
          typename DataType = typename R_type::type>
Scalar<DataType> compute_expected_3x3(const R_type& R, const S_type& S) {
  auto result = make_with_value<Scalar<DataType>>(get<0, 0, 0>(R), 0.0);

  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      for (size_t k = 0; k < Dim; k++) {
        get(result) += R.get(i, j, k) * S.get(i, j, k);
      }
    }
  }

  return result;
}

template <typename A_type, typename B_type,
          typename DataType = typename A_type::type>
Scalar<DataType> compute_expected_4x4(const A_type& A, const B_type& B) {
  auto result = make_with_value<Scalar<DataType>>(get<0, 0, 0, 0>(A), 0.0);

  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      for (size_t k = 0; k < Dim; k++) {
        for (size_t l = 0; l < Dim; l++) {
          get(result) += A.get(i, j, k, l) * B.get(i, j, k, l);
        }
      }
    }
  }

  return result;
}

// \tparam Rank the rank of the operands
template <size_t Rank>
constexpr std::array<size_t, Rank> get_lowest_multi_index_to_sum() {
  return make_array<Rank, size_t>(0);
}

// \tparam Rank the rank of the operands
template <size_t Rank>
constexpr std::array<size_t, Rank> get_highest_multi_index_to_sum() {
  return make_array<Rank, size_t>(Dim - 1);
}

// Note: assumes both operands have same rank with generic indices in same
// order
// \tparam Rank the rank of the operands
template <size_t Rank>
constexpr std::array<size_t, Rank> get_next_lowest_multi_index_to_sum(
    const std::array<size_t, Rank>& uncontracted_multi_index) {
  std::array<size_t, Rank> next_lowest_uncontracted_multi_index =
      uncontracted_multi_index;

  size_t i = Rank - 1;
  while (true) {
    // increment the current index pair's values
    gsl::at(next_lowest_uncontracted_multi_index, i)++;

    // If the index values of the index pair being contracted aren't higher
    // than the maximum values included in the summation, ...
    if (not(gsl::at(next_lowest_uncontracted_multi_index, i) > Dim - 1)) {
      for (size_t j = i + 1; j < Rank; j++) {
        gsl::at(next_lowest_uncontracted_multi_index, j) =
            gsl::at(next_lowest_uncontracted_multi_index, i);
      }

      break;
    }
    // Otherwise, we've wrapped around the highest value being summed over for
    // this index, so we...
    i--;
  }

  return next_lowest_uncontracted_multi_index;
}

// Note: assumes both operands have same rank with generic indices in same
// order
// \tparam Rank the rank of the operands
template <size_t Rank>
constexpr std::array<size_t, Rank> get_next_highest_multi_index_to_sum(
    const std::array<size_t, Rank>& uncontracted_multi_index) {
  std::array<size_t, Rank> next_highest_uncontracted_multi_index =
      uncontracted_multi_index;
  // TODO: assert that this next value is not 0? worth it?
  const size_t max_index_value = gsl::at(uncontracted_multi_index, Rank - 1);

  size_t i = Rank - 1;
  while (i - 1 < Rank and gsl::at(next_highest_uncontracted_multi_index,
                                  i - 1) == max_index_value) {
    if (max_index_value == 1) {
      gsl::at(next_highest_uncontracted_multi_index, i) = max_index_value;
    }
    i--;
  }

  gsl::at(next_highest_uncontracted_multi_index, i)--;

  return next_highest_uncontracted_multi_index;
}

template <size_t Rank>
constexpr size_t get_num_permutations(
    const std::array<size_t, Rank>& multi_index) {
  size_t num_permutations = static_cast<size_t>(factorial(Rank));

  // TODO : maybe at a tparam for if there are spatial-spacetime indices, and if
  // so, start at Dim, instead
  for (size_t i = Dim - 1; i < Dim; i--) {
    const size_t current_index_value_count =
        static_cast<size_t>(alg::count(multi_index, i));
    if (current_index_value_count > 1) {
      num_permutations /=
          static_cast<size_t>(factorial(current_index_value_count));
    }
  }

  return num_permutations;
}

template <size_t Rank, size_t NumIndComponents>
constexpr std::array<size_t, NumIndComponents> get_multipliers() {
  std::array<size_t, NumIndComponents> multipliers{};
  gsl::at(multipliers, 0) = 1;  // lowest multi-index is all 0s, so only 1 combo

  std::array<size_t, Rank> current_multi_index =
      get_lowest_multi_index_to_sum<Rank>();
  for (size_t i = 1; i < NumIndComponents; i++) {
    current_multi_index =
        get_next_lowest_multi_index_to_sum(current_multi_index);
    gsl::at(multipliers, i) = get_num_permutations(current_multi_index);
  }

  return multipliers;
}

template <size_t Rank, typename T1, typename T2,
          typename DataType = typename T1::type>
Scalar<DataType> compute_contraction(const T1& t1, const T2& t2) {
  constexpr size_t num_ind_comp = T1::structure::size();
  constexpr std::array<size_t, num_ind_comp> multipliers =
      get_multipliers<Rank, num_ind_comp>();

  Scalar<DataType> actual_result{};
  const std::array<size_t, Rank> starting_multi_index =
      get_lowest_multi_index_to_sum<Rank>();
  get(actual_result) = gsl::at(multipliers, 0) * t1.get(starting_multi_index) *
                       t2.get(starting_multi_index);

  std::array<size_t, Rank> previous_multi_index = starting_multi_index;
  for (size_t i = 1; i < num_ind_comp; i++) {
    auto current_multi_index =
        get_next_lowest_multi_index_to_sum(previous_multi_index);
    get(actual_result) += gsl::at(multipliers, i) *
                          t1.get(current_multi_index) *
                          t2.get(current_multi_index);
    previous_multi_index = current_multi_index;
  }

  return actual_result;
}

template <typename Generator, typename DataType>
void test(const gsl::not_null<Generator*> generator,
          const DataType& used_for_size) {
  std::uniform_real_distribution<> distribution(0.1, 2.0);

  // upper <1, 1>
  const auto G =
      make_with_random_values<tnsr::II<DataType, Dim, Frame::Inertial>>(
          generator, distribution, used_for_size);
  // lower <1, 1>
  const auto H =
      make_with_random_values<tnsr::ii<DataType, Dim, Frame::Inertial>>(
          generator, distribution, used_for_size);
  // upper <1, 1, 1>
  const auto R = make_with_random_values<
      Tensor<DataType, Symmetry<1, 1, 1>,
             index_list<SpatialIndex<Dim, UpLo::Up, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Up, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Up, Frame::Inertial>>>>(
      generator, distribution, used_for_size);
  // lower <1, 1, 1>
  const auto S = make_with_random_values<
      Tensor<DataType, Symmetry<1, 1, 1>,
             index_list<SpatialIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Lo, Frame::Inertial>>>>(
      generator, distribution, used_for_size);
  // upper <1, 1, 1, 1>
  const auto A = make_with_random_values<
      Tensor<DataType, Symmetry<1, 1, 1, 1>,
             index_list<SpatialIndex<Dim, UpLo::Up, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Up, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Up, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Up, Frame::Inertial>>>>(
      generator, distribution, used_for_size);
  // lower <1, 1, 1, 1>
  const auto B = make_with_random_values<
      Tensor<DataType, Symmetry<1, 1, 1, 1>,
             index_list<SpatialIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Lo, Frame::Inertial>>>>(
      generator, distribution, used_for_size);
  // lower <2, 2, 1, 1>
  const auto C = make_with_random_values<
      Tensor<DataType, Symmetry<2, 2, 1, 1>,
             index_list<SpatialIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpatialIndex<Dim, UpLo::Lo, Frame::Inertial>>>>(
      generator, distribution, used_for_size);

  // symmetric 3x3
  const auto expected_3x3_result = compute_expected_3x3(R, S);
  const Scalar<DataType> actual_3x3_result = compute_contraction<3>(R, S);
  CHECK_ITERABLE_APPROX(actual_3x3_result, expected_3x3_result);

  // symmetric 4x4
  const auto expected_4x4_result = compute_expected_4x4(A, B);
  const Scalar<DataType> actual_4x4_result = compute_contraction<4>(A, B);
  CHECK_ITERABLE_APPROX(actual_4x4_result, expected_4x4_result);
}

template <typename PositionsOfSymmValue, typename CurrentSymmValue,
          typename Iteration, typename SymmValueToFind>
struct get_symm_positions {
  using type = typename std::conditional_t<
      CurrentSymmValue::value == SymmValueToFind::value,
      tmpl::push_back<PositionsOfSymmValue,
                      tmpl::integral_constant<size_t, Iteration::value>>,
      PositionsOfSymmValue>;
};

template <typename SymmetricIndices, typename CurrentSymmValue,
          typename Symmetry>
struct get_symmetric_indices {
  using positions_this_symm_value = tmpl::enumerated_fold<
      Symmetry, tmpl::list<>,
      get_symm_positions<tmpl::_state, tmpl::_element, tmpl::_3,
                         tmpl::pin<CurrentSymmValue>>,
      tmpl::size_t<0>>;

  using type = typename std::conditional_t<
      (tmpl::size<positions_this_symm_value>::value > 1),
      tmpl::push_back<SymmetricIndices, positions_this_symm_value>,
      SymmetricIndices>;
};

template <typename... T>
struct td;

template <typename ExpectedResult, std::int32_t... Symm>
void test_tmpl_symmetry_stuff() {
  using symmetry = tmpl::integral_list<std::int32_t, Symm...>;
  constexpr size_t num_indices = sizeof...(Symm);
  constexpr std::array<std::int32_t, num_indices> symm = {{Symm...}};
  constexpr std::int32_t max_symm_value = *alg::max_element(symm);
  using symm_set =
      tmpl::as_integral_list<tmpl::range<std::int32_t, 1, max_symm_value + 1>>;

  using symmetry_positions = tmpl::fold<
      symm_set, tmpl::list<>,
      get_symmetric_indices<tmpl::_state, tmpl::_element, tmpl::pin<symmetry>>>;

  static_assert(std::is_same<ExpectedResult, symmetry_positions>::value);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Contract",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  test(make_not_null(&generator), std::numeric_limits<double>::signaling_NaN());
  test(make_not_null(&generator),
       DataVector(5, std::numeric_limits<double>::signaling_NaN()));

  test_tmpl_symmetry_stuff<tmpl::list<tmpl::integral_list<size_t, 1, 4, 5>,
                                      tmpl::integral_list<size_t, 0, 3>>,
                           2, 1, 3, 2, 1, 1>();
}
