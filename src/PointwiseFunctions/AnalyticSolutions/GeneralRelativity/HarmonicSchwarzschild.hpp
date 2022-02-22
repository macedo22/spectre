// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <pup.h>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Solutions.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Tags {
template <typename Tag>
struct dt;
}  // namespace Tags
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace gr {
namespace Solutions {
class HarmonicSchwarzschild : public AnalyticSolution<3_st>,
                              public MarkAsAnalyticSolution {
 public:
  struct Mass {
    using type = double;
    static constexpr Options::String help = {"Mass of the black hole"};
    static type lower_bound() { return 0.; }
  };
  struct Center {
    using type = std::array<double, volume_dim>;
    static constexpr Options::String help = {
        "The [x,y,z] center of the black hole"};
  };
  using options = tmpl::list<Mass, Center>;
  static constexpr Options::String help{
      "Black hole in Kerr-Schild coordinates"};

  HarmonicSchwarzschild(double mass, Center::type center,
                        const Options::Context& context = {});

  explicit HarmonicSchwarzschild(CkMigrateMessage* /*unused*/) {}

  HarmonicSchwarzschild() = default;
  HarmonicSchwarzschild(const HarmonicSchwarzschild& /*rhs*/) = default;
  HarmonicSchwarzschild& operator=(const HarmonicSchwarzschild& /*rhs*/) =
      default;
  HarmonicSchwarzschild(HarmonicSchwarzschild&& /*rhs*/) = default;
  HarmonicSchwarzschild& operator=(HarmonicSchwarzschild&& /*rhs*/) = default;
  ~HarmonicSchwarzschild() = default;

  template <typename DataType, typename Frame, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, volume_dim, Frame>& x, double /*t*/,
      tmpl::list<Tags...> /*meta*/) const {
    static_assert(
        tmpl2::flat_all_v<tmpl::list_contains_v<
                tags<DataType, Frame>,
            Tags>...>,
        "At least one of the requested tags is not supported. The requested "
        "tags are listed as template parameters of the `variables` function.");
    IntermediateVars<DataType, Frame> cache(get_size(*x.begin()));
    IntermediateComputer<DataType, Frame> computer(*this, x);
    return {cache.get_var(computer, Tags{})...};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  SPECTRE_ALWAYS_INLINE double mass() const { return mass_; }
  SPECTRE_ALWAYS_INLINE const std::array<double, volume_dim>& center() const {
    return center_;
  }

  struct internal_tags {
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using x_minus_center = ::Tags::TempI<0, 3, Frame, DataType>;
    template <typename DataType>
    using r = ::Tags::TempScalar<1, DataType>;
    template <typename DataType>
    using one_over_r = ::Tags::TempScalar<2, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using x_over_r = ::Tags::TempI<3, 3, Frame, DataType>;
    template <typename DataType>
    using m_over_r = ::Tags::TempScalar<4, DataType>;
    template <typename DataType>
    using sqrt_f_0 = ::Tags::TempScalar<5, DataType>;
    template <typename DataType>
    using f_0 = ::Tags::TempScalar<6, DataType>;
    template <typename DataType>
    using two_m_over_m_plus_r = ::Tags::TempScalar<7, DataType>;
    template <typename DataType>
    using two_m_over_m_plus_r_squared = ::Tags::TempScalar<8, DataType>;
    template <typename DataType>
    using two_m_over_m_plus_r_cubed = ::Tags::TempScalar<9, DataType>;
    template <typename DataType>
    using g_rr = ::Tags::TempScalar<10, DataType>;
    template <typename DataType>
    using one_over_g_rr = ::Tags::TempScalar<11, DataType>;
    template <typename DataType>
    using g_rr_minus_f_0 = ::Tags::TempScalar<12, DataType>;
    template <typename DataType>
    using d_g_rr = ::Tags::TempScalar<13, DataType>;
    template <typename DataType>
    using d_f_0 = ::Tags::TempScalar<14, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using d_f_0_times_x_over_r = ::Tags::TempI<15, 3, Frame, DataType>;
    template <typename DataType>
    using f_1 = ::Tags::TempScalar<16, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using f_1_times_x_over_r = ::Tags::TempI<17, 3, Frame, DataType>;
    template <typename DataType>
    using f_2 = ::Tags::TempScalar<18, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using f_2_times_xxx_over_r_cubed = ::Tags::TempIII<19, 3, Frame, DataType>;
    template <typename DataType>
    using f_3 = ::Tags::TempScalar<20, DataType>;
    template <typename DataType>
    using f_4 = ::Tags::TempScalar<21, DataType>;
  };

  template <typename DataType, typename Frame = ::Frame::Inertial>
  using CachedBuffer = CachedTempBuffer<
      internal_tags::x_minus_center<DataType, Frame>,
      internal_tags::r<DataType>, internal_tags::one_over_r<DataType>,
      internal_tags::x_over_r<DataType, Frame>,
      internal_tags::m_over_r<DataType>, internal_tags::sqrt_f_0<DataType>,
      internal_tags::f_0<DataType>,
      internal_tags::two_m_over_m_plus_r<DataType>,
      internal_tags::two_m_over_m_plus_r_squared<DataType>,
      internal_tags::two_m_over_m_plus_r_cubed<DataType>,
      internal_tags::g_rr<DataType>, internal_tags::one_over_g_rr<DataType>,
      internal_tags::g_rr_minus_f_0<DataType>, internal_tags::d_g_rr<DataType>,
      internal_tags::d_f_0<DataType>,
      internal_tags::d_f_0_times_x_over_r<DataType, Frame>,
      internal_tags::f_1<DataType>,
      internal_tags::f_1_times_x_over_r<DataType, Frame>,
      internal_tags::f_2<DataType>,
      internal_tags::f_2_times_xxx_over_r_cubed<DataType, Frame>,
      internal_tags::f_3<DataType>, internal_tags::f_4<DataType>,
      gr::Tags::Lapse<DataType>, gr::Tags::Shift<3, Frame, DataType>,
      DerivShift<DataType, Frame>, gr::Tags::SpatialMetric<3, Frame, DataType>,
      DerivSpatialMetric<DataType, Frame>,
      ::Tags::dt<gr::Tags::SpatialMetric<3, Frame, DataType>>,
      gr::Tags::DetSpatialMetric<DataType>,
      gr::Tags::InverseSpatialMetric<3, Frame, DataType>>;

  template <typename DataType, typename Frame = ::Frame::Inertial>
  class IntermediateComputer {
   public:
    using CachedBuffer = HarmonicSchwarzschild::CachedBuffer<DataType, Frame>;

    IntermediateComputer(const HarmonicSchwarzschild& solution,
                         const tnsr::I<DataType, 3, Frame>& x);

    void operator()(
        gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_minus_center,
        gsl::not_null<CachedBuffer*> /*cache*/,
        internal_tags::x_minus_center<DataType, Frame> /*meta*/) const;

    void operator()(gsl::not_null<Scalar<DataType>*> r,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::r<DataType> /*meta*/) const;

    void operator()(gsl::not_null<Scalar<DataType>*> one_over_r,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::one_over_r<DataType> /*meta*/) const;

    void operator()(gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_over_r,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::x_over_r<DataType, Frame> /*meta*/) const;

    void operator()(gsl::not_null<Scalar<DataType>*> m_over_r,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::m_over_r<DataType> /*meta*/) const;

    void operator()(gsl::not_null<Scalar<DataType>*> sqrt_f_0,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::sqrt_f_0<DataType> /*meta*/) const;

    void operator()(gsl::not_null<Scalar<DataType>*> f_0,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::f_0<DataType> /*meta*/) const;

    void operator()(
        gsl::not_null<Scalar<DataType>*> two_m_over_m_plus_r,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::two_m_over_m_plus_r<DataType> /*meta*/) const;

    void operator()(
        gsl::not_null<Scalar<DataType>*> two_m_over_m_plus_r_squared,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::two_m_over_m_plus_r_squared<DataType> /*meta*/) const;

    void operator()(
        gsl::not_null<Scalar<DataType>*> two_m_over_m_plus_r_cubed,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::two_m_over_m_plus_r_cubed<DataType> /*meta*/) const;

    void operator()(gsl::not_null<Scalar<DataType>*> g_rr,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::g_rr<DataType> /*meta*/) const;

    void operator()(gsl::not_null<Scalar<DataType>*> one_over_g_rr,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::one_over_g_rr<DataType> /*meta*/) const;

    void operator()(gsl::not_null<Scalar<DataType>*> g_rr_minus_f_0,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::g_rr_minus_f_0<DataType> /*meta*/) const;

    void operator()(gsl::not_null<Scalar<DataType>*> d_g_rr,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::d_g_rr<DataType> /*meta*/) const;

    void operator()(gsl::not_null<Scalar<DataType>*> d_f_0,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::d_f_0<DataType> /*meta*/) const;

    void operator()(
        gsl::not_null<tnsr::I<DataType, 3, Frame>*> d_f_0_times_x_over_r,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::d_f_0_times_x_over_r<DataType, Frame> /*meta*/) const;

    void operator()(gsl::not_null<Scalar<DataType>*> f_1,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::f_1<DataType> /*meta*/) const;

    void operator()(
        gsl::not_null<tnsr::I<DataType, 3, Frame>*> f_1_times_x_over_r,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::f_1_times_x_over_r<DataType, Frame> /*meta*/) const;

    void operator()(gsl::not_null<Scalar<DataType>*> f_2,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::f_2<DataType> /*meta*/) const;

    void operator()(
        gsl::not_null<tnsr::III<DataType, 3, Frame>*>
            f_2_times_xxx_over_r_cubed,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::f_2_times_xxx_over_r_cubed<DataType, Frame> /*meta*/)
        const;

    void operator()(gsl::not_null<Scalar<DataType>*> f_3,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::f_3<DataType> /*meta*/) const;

    void operator()(gsl::not_null<Scalar<DataType>*> f_4,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::f_4<DataType> /*meta*/) const;

    void operator()(gsl::not_null<Scalar<DataType>*> lapse,
                    gsl::not_null<CachedBuffer*> cache,
                    gr::Tags::Lapse<DataType> /*meta*/) const;

    void operator()(gsl::not_null<tnsr::I<DataType, 3, Frame>*> shift,
                    gsl::not_null<CachedBuffer*> cache,
                    gr::Tags::Shift<3, Frame, DataType> /*meta*/) const;

    void operator()(gsl::not_null<tnsr::iJ<DataType, 3, Frame>*> deriv_shift,
                    gsl::not_null<CachedBuffer*> cache,
                    DerivShift<DataType, Frame> /*meta*/) const;

    void operator()(gsl::not_null<tnsr::ii<DataType, 3, Frame>*> spatial_metric,
                    gsl::not_null<CachedBuffer*> cache,
                    gr::Tags::SpatialMetric<3, Frame, DataType> /*meta*/) const;

    void operator()(
        gsl::not_null<tnsr::ijj<DataType, 3, Frame>*> deriv_spatial_metric,
        gsl::not_null<CachedBuffer*> cache,
        DerivSpatialMetric<DataType, Frame> /*meta*/) const;

    void operator()(
        gsl::not_null<tnsr::ii<DataType, 3, Frame>*> dt_spatial_metric,
        gsl::not_null<CachedBuffer*> cache,
        ::Tags::dt<gr::Tags::SpatialMetric<3, Frame, DataType>> /*meta*/) const;

    void operator()(gsl::not_null<Scalar<DataType>*> det_spatial_metric,
                    gsl::not_null<CachedBuffer*> cache,
                    gr::Tags::DetSpatialMetric<DataType> /*meta*/) const;

    void operator()(
        gsl::not_null<tnsr::II<DataType, 3, Frame>*> inverse_spatial_metric,
        gsl::not_null<CachedBuffer*> cache,
        gr::Tags::InverseSpatialMetric<3, Frame, DataType>
        /*meta*/) const;

   private:
    const HarmonicSchwarzschild& solution_;
    const tnsr::I<DataType, 3, Frame>& x_;
  };

  template <typename DataType, typename Frame = ::Frame::Inertial>
  class IntermediateVars : public CachedBuffer<DataType, Frame> {
   public:
    using CachedBuffer = HarmonicSchwarzschild::CachedBuffer<DataType, Frame>;
    using CachedBuffer::CachedBuffer;
    using CachedBuffer::get_var;

    tnsr::i<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        DerivLapse<DataType, Frame> /*meta*/);

    Scalar<DataType> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        ::Tags::dt<gr::Tags::Lapse<DataType>> /*meta*/);

    tnsr::I<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        ::Tags::dt<gr::Tags::Shift<3, Frame, DataType>> /*meta*/);

    Scalar<DataType> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::SqrtDetSpatialMetric<DataType> /*meta*/);

    tnsr::ii<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::ExtrinsicCurvature<3, Frame, DataType> /*meta*/);
  };

 private:
  double mass_{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, volume_dim> center_ =
      make_array<volume_dim>(std::numeric_limits<double>::signaling_NaN());
};

SPECTRE_ALWAYS_INLINE bool operator==(const HarmonicSchwarzschild& lhs,
                                      const HarmonicSchwarzschild& rhs) {
  return lhs.mass() == rhs.mass() and lhs.center() == rhs.center();
}

SPECTRE_ALWAYS_INLINE bool operator!=(const HarmonicSchwarzschild& lhs,
                                      const HarmonicSchwarzschild& rhs) {
  return not(lhs == rhs);
}
}  // namespace Solutions
}  // namespace gr
