// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/HarmonicSchwarzschild.hpp"

#include <cmath>  // IWYU pragma: keep
#include <numeric>
#include <ostream>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/StdHelpers.hpp"

namespace gr::Solutions {

HarmonicSchwarzschild::HarmonicSchwarzschild(
    const double mass, HarmonicSchwarzschild::Center::type center,
    const Options::Context& context)
    : mass_(mass),
      // clang-tidy: do not std::move trivial types.
      center_(std::move(center))  // NOLINT
{
  if (mass_ < 0.0) {
    PARSE_ERROR(context, "Mass must be non-negative. Given mass: " << mass_);
  }
}

void HarmonicSchwarzschild::pup(PUP::er& p) {
  p | mass_;
  p | center_;
}

template <typename DataType, typename Frame>
HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::
    IntermediateComputer(const HarmonicSchwarzschild& solution,
                         const tnsr::I<DataType, 3, Frame>& x)
    : solution_(solution), x_(x) {}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_minus_center,
    const gsl::not_null<CachedBuffer*> /*cache*/,
    internal_tags::x_minus_center<DataType, Frame> /*meta*/) const {
  for (size_t i = 0; i < 3; ++i) {
    x_minus_center->get(i) = gsl::at(x_, i) - gsl::at(solution_.center(), i);
  }
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> r,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::r<DataType> /*meta*/) const {
  // TODO : ask Dr. Lovelace for clarification on eq.
  const auto& x_minus_center =
      cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});

  get(*r) =
      sqrt(square(get<0>(x_minus_center)) + square(get<1>(x_minus_center)) +
           square(get<2>(x_minus_center)));
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> one_over_r,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::one_over_r<DataType> /*meta*/) const {
  const auto& r = cache->get_var(*this, internal_tags::r<DataType, Frame>{});

  get(*one_over_r) = 1.0 / get(r);
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_over_r,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::x_over_r<DataType, Frame> /*meta*/) const {
  const auto& one_over_r =
      cache->get_var(*this, internal_tags::one_over_r<DataType, Frame>{});

  for (size_t i = 0; i < 3; ++i) {
    x_over_r->get(i) = gsl::at(solution_.center(), i) * get(one_over_r);
  }
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> m_over_r,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::m_over_r<DataType> /*meta*/) const {
  const auto& one_over_r =
      cache->get_var(*this, internal_tags::one_over_r<DataType, Frame>{});

  get(*m_over_r) = solution_.mass() * get(one_over_r);
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> one_plus_m_over_r,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::one_plus_m_over_r<DataType> /*meta*/) const {
  const auto& one_over_r =
      cache->get_var(*this, internal_tags::one_over_r<DataType, Frame>{});

  get(*one_plus_m_over_r) = 1.0 + get(one_over_r);
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> f_0,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::f_0<DataType> /*meta*/) const {
  const auto& one_plus_m_over_r = cache->get_var(
      *this, internal_tags::one_plus_m_over_r<DataType, Frame>{});

  get(*f_0) = square(get(one_plus_m_over_r));
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<double>*> two_m,
    const gsl::not_null<CachedBuffer*> /*cache*/,
    internal_tags::two_m<DataType> /*meta*/) const {
  get(*two_m) = 2.0 * solution_.mass();
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> two_m_over_m_plus_r,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::two_m_over_m_plus_r<DataType> /*meta*/) const {
  const auto two_m =
      cache->get_var(*this, internal_tags::two_m<DataType, Frame>{});
  const auto& r = cache->get_var(*this, internal_tags::r<DataType, Frame>{});

  get(*two_m_over_m_plus_r) = two_m / (solution_.mass() + get(r));
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> two_m_over_m_plus_r_squared,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::two_m_over_m_plus_r_squared<DataType> /*meta*/) const {
  const auto& two_m_over_m_plus_r = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r<DataType, Frame>{});

  get(*two_m_over_m_plus_r_squared) = square(get(two_m_over_m_plus_r));
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> two_m_over_m_plus_r_cubed,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::two_m_over_m_plus_r_cubed<DataType> /*meta*/) const {
  const auto& two_m_over_m_plus_r = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r<DataType, Frame>{});
  const auto& two_m_over_m_plus_r_squared = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r_squared<DataType, Frame>{});

  get(*two_m_over_m_plus_r_cubed) =
      get(two_m_over_m_plus_r) * get(two_m_over_m_plus_r_squared);
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<double>*> one_over_m,
    const gsl::not_null<CachedBuffer*> /*cache*/,
    internal_tags::two_m<DataType> /*meta*/) const {
  get(*one_over_m) = 1.0 / solution_.mass();
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> g_rr,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::g_rr<DataType> /*meta*/) const {
  const auto& two_m_over_m_plus_r = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r<DataType, Frame>{});
  const auto& two_m_over_m_plus_r_squared = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r_squared<DataType, Frame>{});
  const auto& two_m_over_m_plus_r_cubed = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r_cubed<DataType, Frame>{});

  get(*g_rr) = 1.0 + get(two_m_over_m_plus_r) +
               get(two_m_over_m_plus_r_squared) +
               get(two_m_over_m_plus_r_cubed);
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> one_over_grr,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::one_over_grr<DataType> /*meta*/) const {
  const auto& grr =
      cache->get_var(*this, internal_tags::grr<DataType, Frame>{});

  get(*one_over_grr) = 1.0 / get(grr);
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> g_rr_minus_f_0,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::g_rr_minus_f_0<DataType> /*meta*/) const {
  const auto& grr =
      cache->get_var(*this, internal_tags::grr<DataType, Frame>{});
  const auto& f_0 =
      cache->get_var(*this, internal_tags::f_0<DataType, Frame>{});

  get(*g_rr_minus_f_0) = get(grr) - get(f_0);
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> d_g_rr,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::d_g_rr<DataType> /*meta*/) const {
  const auto one_over_m =
      cache->get_var(*this, internal_tags::one_over_m<DataType, Frame>{});
  const auto& two_m_over_m_plus_r = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r<DataType, Frame>{});
  const auto& two_m_over_m_plus_r_squared = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r_squared<DataType, Frame>{});
  const auto& two_m_over_m_plus_r_cubed = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r_cubed<DataType, Frame>{});

  get(*d_g_rr) =
      -one_over_m *
      (0.5 * get(two_m_over_m_plus_r_squared) + get(two_m_over_m_plus_r_cubed) +
       1.5 * get(two_m_over_m_plus_r) * get(two_m_over_m_plus_r_cubed));
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> d_g_rr,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::d_g_rr<DataType> /*meta*/) const {
  const auto one_over_m =
      cache->get_var(*this, internal_tags::one_over_m<DataType, Frame>{});
  const auto& two_m_over_m_plus_r = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r<DataType, Frame>{});
  const auto& two_m_over_m_plus_r_squared = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r_squared<DataType, Frame>{});
  const auto& two_m_over_m_plus_r_cubed = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r_cubed<DataType, Frame>{});

  get(*d_g_rr) =
      -one_over_m *
      (0.5 * get(two_m_over_m_plus_r_squared) + get(two_m_over_m_plus_r_cubed) +
       1.5 * get(two_m_over_m_plus_r) * get(two_m_over_m_plus_r_cubed));
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> d_f_0,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::d_g_rr<DataType> /*meta*/) const {
  const auto& f_0 =
      cache->get_var(*this, internal_tags::f_0<DataType, Frame>{});
  const auto& m_over_r =
      cache->get_var(*this, internal_tags::m_over_r<DataType, Frame>{});
  const auto& one_over_r =
      cache->get_var(*this, internal_tags::one_over_r<DataType, Frame>{});

  get(*d_f_0) = -2.0 * get(f_0) * get(m_over_r) * get(one_over_r);
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> d_f_0_times_x_over_r,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::d_f_0_times_x_over_r<DataType, Frame> /*meta*/) const {
  const auto& d_f_0 =
      cache->get_var(*this, internal_tags::d_f_0<DataType, Frame>{});
  const auto& x_over_r =
      cache->get_var(*this, internal_tags::x_over_r<DataType, Frame>{});

  get(*d_f_0_times_x_over_r)<0> = get(d_f_0) * get<0>(x_over_r);
  get(*d_f_0_times_x_over_r)<1> = get(d_f_0) * get<1>(x_over_r);
  get(*d_f_0_times_x_over_r)<2> = get(d_f_0) * get<2>(x_over_r);
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> f_1,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::f_1<DataType> /*meta*/) const {
  const auto& one_over_r =
      cache->get_var(*this, internal_tags::one_over_r<DataType, Frame>{});
  const auto& grr =
      cache->get_var(*this, internal_tags::grr<DataType, Frame>{});
  const auto& f_0 =
      cache->get_var(*this, internal_tags::f_0<DataType, Frame>{});

  get(*f_1) = get(one_over_r) * (get(grr) * get(f_0));
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> f_1_times_x_over_r,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::f_1_times_x_over_r<DataType, Frame> /*meta*/) const {
  const auto& f_1 =
      cache->get_var(*this, internal_tags::f_1<DataType, Frame>{});
  const auto& x_over_r =
      cache->get_var(*this, internal_tags::x_over_r<DataType, Frame>{});

  get(*f_1_times_x_over_r)<0> = get(f_1) * get<0>(x_over_r);
  get(*f_1_times_x_over_r)<1> = get(f_1) * get<1>(x_over_r);
  get(*f_1_times_x_over_r)<2> = get(f_1) * get<2>(x_over_r);
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> f_2,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::f_2<DataType> /*meta*/) const {
  const auto& d_grr =
      cache->get_var(*this, internal_tags::d_grr<DataType, Frame>{});
  const auto& d_f_0 =
      cache->get_var(*this, internal_tags::d_f_0<DataType, Frame>{});
  const auto& f_2 =
      cache->get_var(*this, internal_tags::f_2<DataType, Frame>{});

  get(*f_2) = get(d_grr) - get(d_f_0) - 2.0 * get(f_1);
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::III<DataType, 3, Frame>*>
        f_2_times_xxx_over_r_cubed,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::xxx_over_r_cubed<DataType, Frame> /*meta*/) const {
  const auto& f_2 =
      cache->get_var(*this, internal_tags::f_2<DataType, Frame>{});
  const auto& x_over_r =
      cache->get_var(*this, internal_tags::x_over_r<DataType, Frame>{});

  get<0, 0, 0>(f_2_times_xxx_over_r_cubed) =
      get(f_2) * get<0>(x_over_r) * get<1>(x_over_r) * get<2>(x_over_r);
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> f_3,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::f_3<DataType> /*meta*/) const {
  const auto& one_over_r =
      cache->get_var(*this, internal_tags::one_over_r<DataType, Frame>{});
  const auto& two_m_over_m_plus_r_squared = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r_squared<DataType, Frame>{});
  const auto& grr =
      cache->get_var(*this, internal_tags::grr<DataType, Frame>{});

  get(*f_3) = get(one_over_r) * get(two_m_over_m_plus_r_squared) / get(grr);
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> f_4,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::f_4<DataType> /*meta*/) const {
  const auto& f_3 =
      cache->get_var(*this, internal_tags::f_3<DataType, Frame>{});
  const auto one_over_m =
      cache->get_var(*this, internal_tags::one_over_m<DataType, Frame>{});
  const auto& two_m_over_m_plus_r_squared = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r_squared<DataType, Frame>{});
  const auto& two_m_over_m_plus_r_cubed = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r_cubed<DataType, Frame>{});
  const auto& grr =
      cache->get_var(*this, internal_tags::grr<DataType, Frame>{});
  const auto& d_grr =
      cache->get_var(*this, internal_tags::d_grr<DataType, Frame>{});
  const auto& one_over_grr =
      cache->get_var(*this, internal_tags::one_over_grr<DataType, Frame>{});

  get(*f_4) =
      -get(f_3) - one_over_m * get(two_m_over_m_plus_r_cubed) / get(grr) -
      get(d_grr) * get(two_m_over_m_plus_r_squared) * square(get(one_over_grr));
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> lapse,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::Lapse<DataType> /*meta*/) const {
  const auto& one_over_grr =
      cache->get_var(*this, internal_tags::one_over_grr<DataType, Frame>{});
  get(*lapse) = sqrt(one_over_grr);
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> shift,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::Shift<3, Frame, DataType> /*meta*/) const {
  // TODO : ask about this eq - unequal indices? should TEs allow this?
  const auto& two_m_over_m_plus_r = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r<DataType, Frame>{});
  const auto& one_over_grr =
      cache->get_var(*this, internal_tags::one_over_grr<DataType, Frame>{});

  get<0>(*shift) = get(two_m_over_m_plus_r) * get(one_over_grr);
  get<1>(*shift) = get<0>(*shift);
  get<2>(*shift) = get<0>(*shift);
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::iJ<DataType, 3, Frame>*> deriv_shift,
    const gsl::not_null<CachedBuffer*> cache,
    DerivShift<DataType, Frame> /*meta*/) const {
  const auto& x_over_r =
      cache->get_var(*this, internal_tags::x_over_r<DataType, Frame>{});
  const auto& f_3 =
      cache->get_var(*this, internal_tags::f_3<DataType, Frame>{});
  const auto& f_4 =
      cache->get_var(*this, internal_tags::f_4<DataType, Frame>{});

  for (size_t k = 0; k < 3; ++k) {
    for (size_t i = k; i < 3; ++i) {
      if (i != k) {
        deriv_shift->get(k, i) = get(f_4) * x_over_r.get(i) * x_over_r.get(k);
        deriv_shift->get(i, k) = deriv_shift->get(k, i);
      } else {
        deriv_shift->get(k, i) = get(f_4) * square(x_over_r.get(i)) + 1.0;
      }
    }
  }
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> spatial_metric,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::SpatialMetric<3, Frame, DataType> /*meta*/) const {
  const auto& f_0 =
      cache->get_var(*this, internal_tags::f_0<DataType, Frame>{});
  const auto& g_rr_minus_f_0 =
      cache->get_var(*this, internal_tags::g_rr_minus_f_0<DataType, Frame>{});
  const auto& x_over_r =
      cache->get_var(*this, internal_tags::x_over_r<DataType, Frame>{});

  for (size_t i = 0; i < 3; ++k) {
    for (size_t j = i; i < 3; ++j) {
      spatial_metric->get(i, j) =
          get(g_rr_minus_f_0) * x_over_r.get(i) * x_over_r.get(j);
      if (i == j) {
        spatial_metric->get(i, j) += get(f_0);
      }
    }
  }
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, 3, Frame>*> deriv_spatial_metric,
    const gsl::not_null<CachedBuffer*> cache,
    DerivSpatialMetric<DataType, Frame> /*meta*/) const {
  const auto& d_f_0_times_x_over_r = cache->get_var(
      *this, internal_tags::d_f_0_times_x_over_r<DataType, Frame>{});
  const auto& f_1_times_x_over_r = cache->get_var(
      *this, internal_tags::f_1_times_x_over_r<DataType, Frame>{});
  const auto& f_2_times_xxx_over_r_cubed = cache->get_var(
      *this, internal_tags::f_2_times_xxx_over_r_cubed<DataType, Frame>{});

  for (size_t k = 0; k < 3; ++k) {
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        deriv_spatial_metric->get(k, i, j) = get(f_2_times_xxx_over_r_cubed);
        if (i == j) {
          deriv_spatial_metric->get(k, i, j) += d_f_0_times_x_over_r.get(k);
        }
        if (j == k) {
          deriv_spatial_metric->get(k, i, j) += f_1_times_x_over_r.get(i);
        }
        if (i == k) {
          deriv_spatial_metric->get(k, i, j) += f_1_times_x_over_r.get(j);
        }
      }
    }
  }
}

template <typename DataType, typename Frame>
void HarmonicSchwarzschild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> dt_spatial_metric,
    const gsl::not_null<CachedBuffer*> /*cache*/,
    ::Tags::dt<gr::Tags::SpatialMetric<3, Frame, DataType>> /*meta*/) const {
  std::fill(dt_spatial_metric->begin(), dt_spatial_metric->end(), 0.);
}

template <typename DataType, typename Frame>
Scalar<DataType>
HarmonicSchwarzschild::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    ::Tags::dt<gr::Tags::Lapse<DataType>> /*meta*/) {
  const auto& r = get(get_var(computer, internal_tags::r<DataType>{}));
  return make_with_value<Scalar<DataType>>(r, 0.);
}

template <typename DataType, typename Frame>
tnsr::I<DataType, 3, Frame>
HarmonicSchwarzschild::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    ::Tags::dt<gr::Tags::Shift<3, Frame, DataType>> /*meta*/) {
  const auto& r = get(get_var(computer, internal_tags::r<DataType>{}));
  return make_with_value<Scalar<DataType>>(r, 0.);
}

template <typename DataType, typename Frame>
Scalar<DataType>
HarmonicSchwarzschild::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    gr::Tags::SqrtDetSpatialMetric<DataType> /*meta*/) {
  // TODO : ask if there is an eq I should be implementing instead
  return Scalar<DataType>(get(sqrt(determinant(
      get_var(computer, gr::Tags::SpatialMetric<DataType, Frame>{})))));
}

template <typename DataType, typename Frame>
tnsr::II<DataType, 3, Frame>
HarmonicSchwarzschild::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    gr::Tags::InverseSpatialMetric<3, Frame, DataType> /*meta*/) {
  // TODO : store det_and_inverse instead?
}

template <typename DataType, typename Frame>
tnsr::ii<DataType, 3, Frame>
HarmonicSchwarzschild::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    gr::Tags::ExtrinsicCurvature<3, Frame, DataType> /*meta*/) {
  return gr::extrinsic_curvature(
      get_var(computer, gr::Tags::Lapse<DataType>{}),
      get_var(computer, gr::Tags::Shift<3, Frame, DataType>{}),
      get_var(computer, DerivShift<DataType, Frame>{}),
      get_var(computer, gr::Tags::SpatialMetric<3, Frame, DataType>{}),
      get_var(computer,
              ::Tags::dt<gr::Tags::SpatialMetric<3, Frame, DataType>>{}),
      get_var(computer, DerivSpatialMetric<DataType, Frame>{}));
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                              \
  template class HarmonicSchwarzschild::IntermediateVars<DTYPE(data),     \
                                                         FRAME(data)>;    \
  template class HarmonicSchwarzschild::IntermediateComputer<DTYPE(data), \
                                                             FRAME(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector, double),
                        (::Frame::Inertial, ::Frame::Grid))
#undef INSTANTIATE
#undef DTYPE
#undef FRAME
}  // namespace gr::Solutions
