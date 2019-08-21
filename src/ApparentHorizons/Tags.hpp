// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "ApparentHorizons/ComputeItems.hpp"
#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/StrahlkorperGr.hpp"
#include "ApparentHorizons/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "ApparentHorizons/TagsTypeAliases.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare gr::Tags::SpatialMetric
/// \cond
class DataVector;
class FastFlow;
/// \endcond

namespace ah {
namespace Tags {
struct FastFlow : db::SimpleTag {
  static std::string name() noexcept { return "FastFlow"; }
  using type = ::FastFlow;
};
}  // namespace Tags
}  // namespace ah

/// \ingroup SurfacesGroup
/// Holds tags and ComputeItems associated with a `::Strahlkorper`.
namespace StrahlkorperTags {

/// Tag referring to a `::Strahlkorper`
template <typename Frame>
struct Strahlkorper : db::SimpleTag {
  static std::string name() noexcept { return "Strahlkorper"; }
  using type = ::Strahlkorper<Frame>;
};

struct YlmSpherepack : db::SimpleTag {
  static std::string name() noexcept { return "YlmSpherepack"; }
  using type = ::YlmSpherepack;
};

template <typename Frame>
struct YlmSpherepackCompute : YlmSpherepack, db::ComputeTag {
  static const ::YlmSpherepack& function(
      const ::Strahlkorper<Frame>& strahlkorper) noexcept {
          return strahlkorper.ylm_spherepack();
      }
  using argument_tags = tmpl::list<Strahlkorper<Frame>>;
};

/// \f$(\theta,\phi)\f$ on the grid.
/// Doesn't depend on the shape of the surface.
template <typename Frame>
struct ThetaPhi : db::ComputeTag {
  static std::string name() noexcept { return "ThetaPhi"; }
  static aliases::ThetaPhi<Frame> function(
      const ::Strahlkorper<Frame>& strahlkorper) noexcept;
  using argument_tags = tmpl::list<Strahlkorper<Frame>>;
};

/// `Rhat(i)` is \f$\hat{r}^i = x_i/\sqrt{x^2+y^2+z^2}\f$ on the grid.
/// Doesn't depend on the shape of the surface.
template <typename Frame>
struct Rhat : db::ComputeTag {
  static std::string name() noexcept { return "Rhat"; }
  static aliases::OneForm<Frame> function(
      const db::item_type<ThetaPhi<Frame>>& theta_phi) noexcept;
  using argument_tags = tmpl::list<ThetaPhi<Frame>>;
};

/// `Jacobian(i,0)` is \f$\frac{1}{r}\partial x^i/\partial\theta\f$,
/// and `Jacobian(i,1)`
/// is \f$\frac{1}{r\sin\theta}\partial x^i/\partial\phi\f$.
/// Here \f$r\f$ means \f$\sqrt{x^2+y^2+z^2}\f$.
/// `Jacobian` doesn't depend on the shape of the surface.
template <typename Frame>
struct Jacobian : db::ComputeTag {
  static std::string name() noexcept { return "Jacobian"; }
  static aliases::Jacobian<Frame> function(
      const db::item_type<ThetaPhi<Frame>>& theta_phi) noexcept;
  using argument_tags = tmpl::list<ThetaPhi<Frame>>;
};

/// `InvJacobian(0,i)` is \f$r\partial\theta/\partial x^i\f$,
/// and `InvJacobian(1,i)` is \f$r\sin\theta\partial\phi/\partial x^i\f$.
/// Here \f$r\f$ means \f$\sqrt{x^2+y^2+z^2}\f$.
/// `InvJacobian` doesn't depend on the shape of the surface.
template <typename Frame>
struct InvJacobian : db::ComputeTag {
  static std::string name() noexcept { return "InvJacobian"; }
  static aliases::InvJacobian<Frame> function(
      const db::item_type<ThetaPhi<Frame>>& theta_phi) noexcept;
  using argument_tags = tmpl::list<ThetaPhi<Frame>>;
};

/// `InvHessian(k,i,j)` is \f$\partial (J^{-1}){}^k_j/\partial x^i\f$,
/// where \f$(J^{-1}){}^k_j\f$ is the inverse Jacobian.
/// `InvHessian` is not symmetric because the Jacobians are Pfaffian.
/// `InvHessian` doesn't depend on the shape of the surface.
template <typename Frame>
struct InvHessian : db::ComputeTag {
  static std::string name() noexcept { return "InvHessian"; }
  static aliases::InvHessian<Frame> function(
      const db::item_type<ThetaPhi<Frame>>& theta_phi) noexcept;
  using argument_tags = tmpl::list<ThetaPhi<Frame>>;
};

/// (Euclidean) distance \f$r_{\rm surf}(\theta,\phi)\f$ from the center to each
/// point of the surface.
template <typename Frame>
struct Radius : db::ComputeTag {
  static std::string name() noexcept { return "Radius"; }
  SPECTRE_ALWAYS_INLINE static auto function(
      const ::Strahlkorper<Frame>& strahlkorper) noexcept {
    return strahlkorper.ylm_spherepack().spec_to_phys(
        strahlkorper.coefficients());
  }
  using argument_tags = tmpl::list<Strahlkorper<Frame>>;
};

/// `CartesianCoords(i)` is \f$x_{\rm surf}^i\f$,
/// the vector of \f$(x,y,z)\f$ coordinates of each point
/// on the surface.
template <typename Frame>
struct CartesianCoords : db::ComputeTag {
  static std::string name() noexcept { return "CartesianCoords"; }
  static aliases::Vector<Frame> function(
      const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
      const db::item_type<Rhat<Frame>>& r_hat) noexcept;
  using argument_tags =
      tmpl::list<Strahlkorper<Frame>, Radius<Frame>, Rhat<Frame>>;
};

/// `DxRadius(i)` is \f$\partial r_{\rm surf}/\partial x^i\f$.  Here
/// \f$r_{\rm surf}=r_{\rm surf}(\theta,\phi)\f$ is the function
/// describing the surface, which is considered a function of
/// Cartesian coordinates
/// \f$r_{\rm surf}=r_{\rm surf}(\theta(x,y,z),\phi(x,y,z))\f$
/// for this operation.
template <typename Frame>
struct DxRadius : db::ComputeTag {
  static std::string name() noexcept { return "DxRadius"; }
  static aliases::OneForm<Frame> function(
      const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
      const db::item_type<InvJacobian<Frame>>& inv_jac) noexcept;
  using argument_tags =
      tmpl::list<Strahlkorper<Frame>, Radius<Frame>, InvJacobian<Frame>>;
};

/// `D2xRadius(i,j)` is
/// \f$\partial^2 r_{\rm surf}/\partial x^i\partial x^j\f$. Here
/// \f$r_{\rm surf}=r_{\rm surf}(\theta,\phi)\f$ is the function
/// describing the surface, which is considered a function of
/// Cartesian coordinates
/// \f$r_{\rm surf}=r_{\rm surf}(\theta(x,y,z),\phi(x,y,z))\f$
/// for this operation.
template <typename Frame>
struct D2xRadius : db::ComputeTag {
  static std::string name() noexcept { return "D2xRadius"; }
  static aliases::SecondDeriv<Frame> function(
      const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
      const db::item_type<InvJacobian<Frame>>& inv_jac,
      const db::item_type<InvHessian<Frame>>& inv_hess) noexcept;
  using argument_tags = tmpl::list<Strahlkorper<Frame>, Radius<Frame>,
                                   InvJacobian<Frame>, InvHessian<Frame>>;
};

/// \f$\nabla^2 r_{\rm surf}\f$, the flat Laplacian of the surface.
/// This is \f$\eta^{ij}\partial^2 r_{\rm surf}/\partial x^i\partial x^j\f$,
/// where \f$r_{\rm surf}=r_{\rm surf}(\theta(x,y,z),\phi(x,y,z))\f$.
template <typename Frame>
struct LaplacianRadius : db::ComputeTag {
  static std::string name() noexcept { return "LaplacianRadius"; }
  static DataVector function(
      const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
      const db::item_type<ThetaPhi<Frame>>& theta_phi) noexcept;
  using argument_tags =
      tmpl::list<Strahlkorper<Frame>, Radius<Frame>, ThetaPhi<Frame>>;
};

/// `NormalOneForm(i)` is \f$s_i\f$, the (unnormalized) normal one-form
/// to the surface, expressed in Cartesian components.
/// This is computed by \f$x_i/r-\partial r_{\rm surf}/\partial x^i\f$,
/// where \f$x_i/r\f$ is `Rhat` and
/// \f$\partial r_{\rm surf}/\partial x^i\f$ is `DxRadius`.
/// See Eq. (8) of \cite Baumgarte1996hh.
/// Note on the word "normal": \f$s_i\f$ points in the correct direction
/// (it is "normal" to the surface), but it does not have unit length
/// (it is not "normalized"; normalization requires a metric).
template <typename Frame>
struct NormalOneForm : db::ComputeTag {
  static std::string name() noexcept { return "NormalOneForm"; }
  static aliases::OneForm<Frame> function(
      const db::item_type<DxRadius<Frame>>& dx_radius,
      const db::item_type<Rhat<Frame>>& r_hat) noexcept;
  using argument_tags = tmpl::list<DxRadius<Frame>, Rhat<Frame>>;
};

/// `Tangents(i,j)` is \f$\partial x_{\rm surf}^i/\partial q^j\f$,
/// where \f$x_{\rm surf}^i\f$ are the Cartesian coordinates of the
/// surface (i.e. `CartesianCoords`) and are considered functions of
/// \f$(\theta,\phi)\f$.
///
/// \f$\partial/\partial q^0\f$ means
/// \f$\partial/\partial\theta\f$; and \f$\partial/\partial q^1\f$
/// means \f$\csc\theta\,\,\partial/\partial\phi\f$.  Note that the
/// vectors `Tangents(i,0)` and `Tangents(i,1)` are orthogonal to the
/// `NormalOneForm` \f$s_i\f$, i.e.
/// \f$s_i \partial x_{\rm surf}^i/\partial q^j = 0\f$; this statement
/// is independent of a metric.  Also, `Tangents(i,0)` and
/// `Tangents(i,1)` are not necessarily orthogonal to each other,
/// since orthogonality between 2 vectors (as opposed to a vector and
/// a one-form) is metric-dependent.
template <typename Frame>
struct Tangents : db::ComputeTag {
  static std::string name() noexcept { return "Tangents"; }
  static aliases::Jacobian<Frame> function(
      const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
      const db::item_type<Rhat<Frame>>& r_hat,
      const db::item_type<Jacobian<Frame>>& jac) noexcept;
  using argument_tags = tmpl::list<Strahlkorper<Frame>, Radius<Frame>,
                                   Rhat<Frame>, Jacobian<Frame>>;
};

/// Computes the Euclidean area element on a Strahlkorper.
/// Useful for flat space integrals.
template <typename Frame>
struct EuclideanAreaElement : db::ComputeTag {
  static std::string name() noexcept { return "EuclideanAreaElement"; }
  static constexpr auto function =
      ::StrahlkorperGr::euclidean_area_element<Frame>;
  using argument_tags = tmpl::list<
      StrahlkorperTags::Jacobian<Frame>, StrahlkorperTags::NormalOneForm<Frame>,
      StrahlkorperTags::Radius<Frame>, StrahlkorperTags::Rhat<Frame>>;
};

/// Computes the flat-space integral of a scalar over a Strahlkorper.
template <typename IntegrandTag, typename Frame>
struct EuclideanSurfaceIntegral : db::ComputeTag {
  static std::string name() noexcept {
    return "EuclideanSurfaceIntegral" + IntegrandTag::name();
  }
  static constexpr auto function =
      ::StrahlkorperGr::surface_integral_of_scalar<Frame>;
  using argument_tags = tmpl::list<EuclideanAreaElement<Frame>, IntegrandTag,
                                   StrahlkorperTags::Strahlkorper<Frame>>;
};

/// Computes the Euclidean-space integral of a vector over a
/// Strahlkorper, \f$\oint V^i s_i (s_j s_k \delta^{jk})^{-1/2} d^2 S\f$,
/// where \f$s_i\f$ is the Strahlkorper surface unit normal and
/// \f$\delta^{ij}\f$ is the Kronecker delta.  Note that \f$s_i\f$ is
/// not assumed to be normalized; the denominator of the integrand
/// effectively normalizes it using the Euclidean metric.
template <typename IntegrandTag, typename Frame>
struct EuclideanSurfaceIntegralVector : db::ComputeTag {
  static std::string name() noexcept {
    return "EuclideanSurfaceIntegralVector(" + IntegrandTag::name() + ")";
  }
  static constexpr auto function =
      ::StrahlkorperGr::euclidean_surface_integral_of_vector<Frame>;
  using argument_tags = tmpl::list<EuclideanAreaElement<Frame>, IntegrandTag,
                                   StrahlkorperTags::NormalOneForm<Frame>,
                                   StrahlkorperTags::Strahlkorper<Frame>>;
};

template <typename Frame>
using items_tags = tmpl::list<Strahlkorper<Frame>>;

template <typename Frame>
using compute_items_tags =
    tmpl::list<ThetaPhi<Frame>, Rhat<Frame>, Jacobian<Frame>,
               InvJacobian<Frame>, InvHessian<Frame>, Radius<Frame>,
               CartesianCoords<Frame>, DxRadius<Frame>, D2xRadius<Frame>,
               LaplacianRadius<Frame>, NormalOneForm<Frame>, Tangents<Frame>>;

}  // namespace StrahlkorperTags

namespace StrahlkorperGr {
/// \ingroup SurfacesGroup
/// Holds tags and ComputeItems associated with a `::Strahlkorper` that
/// also need a metric.
namespace Tags {

/// Computes one over the magnitude of \f$s^j\f$.

struct OneOverOneFormMagnitude : db::SimpleTag {
  static std::string name() noexcept { return "OneOverOneFormMagnitude"; }
  using type = DataVector;
};

template <typename Frame>
struct OneOverOneFormMagnitudeCompute : OneOverOneFormMagnitude,
                                        db::ComputeTag {
  static DataVector function(
      const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric,
      const tnsr::i<DataVector, 3, Frame>& normal_one_form) noexcept {
    return 1.0 / get(magnitude(normal_one_form, inverse_spatial_metric));
  }
  using argument_tags =
      tmpl::list<gr::Tags::InverseSpatialMetric<3, Frame, DataVector>,
                 StrahlkorperTags::NormalOneForm<Frame>>;
};

/// See Eqs. (1--9) of \cite Baumgarte1996hh.
/// Here \f$S_j\f$ is the (normalized) unit one-form to the surface,
/// and \f$D_i\f$ is the spatial covariant derivative.  Note that this
/// object is symmetric, even though this is not obvious from the
/// definition.  The input arguments `r_hat`, `radius`, and
/// `d2x_radius` depend on the Strahlkorper but not on the metric, and
/// can be computed from a Strahlkorper using ComputeItems in
/// `StrahlkorperTags`.  The input argument
/// `one_over_one_form_magnitude` is \f$1/\sqrt{g^{ij}n_i n_j}\f$,
/// where \f$n_i\f$ is `StrahlkorperTags::NormalOneForm` (i.e.  the
/// unnormalized one-form to the Strahlkorper); it can be computed
/// using (one over) the `magnitude` function.  The input argument
/// `unit_normal_one_form` is \f$S_j\f$,the normalized one-form.

template <typename Frame>
struct UnitNormalOneForm : db::SimpleTag {
  static std::string name() noexcept { return "UnitNormalOneForm"; }
  using type = tnsr::i<DataVector, 3, Frame>;
};

template <typename Frame>
struct UnitNormalOneFormCompute : UnitNormalOneForm<Frame>, db::ComputeTag {
  static constexpr auto function = &StrahlkorperGr::unit_normal_one_form<Frame>;
  using argument_tags =
      tmpl::list<StrahlkorperTags::NormalOneForm<Frame>,
                 OneOverOneFormMagnitude>;
};
/// UnitNormalVector is defined as f$S^i = g^{ij} S_j\f$,
/// where \f$S_j\f$ is the unit normal one form.

template <typename Frame>
struct UnitNormalVector : db::SimpleTag {
  static std::string name() noexcept { return "UnitNormalVector"; }
  using type = tnsr::I<DataVector, 3, Frame>;
};

template <typename Frame>
struct UnitNormalVectorCompute : UnitNormalVector<Frame>, db::ComputeTag {
  static tnsr::I<DataVector, 3, Frame> function(
      const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric,
      const tnsr::i<DataVector, 3, Frame>& unit_normal_one_form) noexcept {
    return raise_or_lower_index(unit_normal_one_form, inverse_spatial_metric);
  }
  using argument_tags =
      tmpl::list<gr::Tags::InverseSpatialMetric<3, Frame, DataVector>,
                 UnitNormalOneForm<Frame>>;
};

template <typename Frame>
struct GradUnitNormalOneForm : db::SimpleTag {
  static std::string name() noexcept { return "GradUnitNormalOneForm"; }
  using type = tnsr::ii<DataVector, 3, Frame>;
};

template <typename Frame>
struct GradUnitNormalOneFormCompute : GradUnitNormalOneForm<Frame>,
                                      db::ComputeTag {
  static constexpr auto function =
      &StrahlkorperGr::grad_unit_normal_one_form<Frame>;
  using argument_tags =
      tmpl::list<StrahlkorperTags::Rhat<Frame>,
                 StrahlkorperTags::Radius<Frame>, UnitNormalOneForm<Frame>,
                 StrahlkorperTags::D2xRadius<Frame>, OneOverOneFormMagnitude,
                 gr::Tags::SpatialChristoffelSecondKind<3, Frame, DataVector>>;
};


template <typename Frame>
struct InverseSurfaceMetric : db::SimpleTag {
  using type = tnsr::II<DataVector, 3, Frame>;
  static std::string name() noexcept { return "InverseSurfaceMetric"; }
};

template <typename Frame>
struct InverseSurfaceMetricCompute : InverseSurfaceMetric<Frame>,
                                     db::ComputeTag {
  static constexpr auto function =
      &StrahlkorperGr::inverse_surface_metric<Frame>;
  using argument_tags =
      tmpl::list<UnitNormalVector<Frame>,
                 gr::Tags::InverseSpatialMetric<3, Frame, DataVector>>;
};

struct Expansion : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Expansion"; }
};

template <typename Frame>
struct ExpansionCompute : Expansion, db::ComputeTag {
  static constexpr auto function = &StrahlkorperGr::expansion<Frame>;
  using argument_tags =
      tmpl::list<GradUnitNormalOneForm<Frame>,
                 InverseSurfaceMetric<Frame>,
                 gr::Tags::ExtrinsicCurvature<3, Frame, DataVector>>;
};

template <typename Frame>
struct ExtrinsicCurvature : db::SimpleTag {
  using type = tnsr::ii<DataVector, 3, Frame>;
  static std::string name() noexcept { return "ExtrinsicCurvature"; }
};

template <typename Frame>
struct ExtrinsicCurvatureCompute : ExtrinsicCurvature<Frame>, db::ComputeTag {
  static constexpr auto function = &StrahlkorperGr::extrinsic_curvature<Frame>;
  using argument_tags =
      tmpl::list<GradUnitNormalOneForm<Frame>, UnitNormalOneForm<Frame>,
                 UnitNormalVector<Frame>>;
};

struct HorizonRicciScalar : db::SimpleTag {
  static std::string name() noexcept { return "HorizonRicciScalar"; }
  using type = Scalar<DataVector>;
};

template <typename Frame>
struct HorizonRicciScalarCompute : HorizonRicciScalar, db::ComputeTag {
  static constexpr auto function = &StrahlkorperGr::ricci_scalar<Frame>;
  using argument_tags =
      tmpl::list<gr::Tags::RicciTensor<3, Frame, DataVector>,
                 UnitNormalVector<Frame>, ExtrinsicCurvature<Frame>,
                 gr::Tags::InverseSpatialMetric<3, Frame, DataVector>>;
};

struct MaxHorizonRicciScalar : db::SimpleTag {
  static std::string name() noexcept { return "MaxHorizonRicciScalar"; }
  using type = double;
};

struct MaxHorizonRicciScalarCompute : MaxHorizonRicciScalar, db::ComputeTag {
  static double function(const Scalar<DataVector>& ricci_scalar) {
    return max(get(ricci_scalar));
  }
  using argument_tags = tmpl::list<HorizonRicciScalar>;
};

struct MinHorizonRicciScalar : db::SimpleTag {
  static std::string name() noexcept { return "MinHorizonRicciScalar"; }
  using type = double;
};

struct MinHorizonRicciScalarCompute : MinHorizonRicciScalar, db::ComputeTag {
  static double function(const Scalar<DataVector>& ricci_scalar) {
    return min(get(ricci_scalar));
  }
  using argument_tags = tmpl::list<HorizonRicciScalar>;
};

struct SpatialRicciScalar : db::SimpleTag {
  static std::string name() noexcept { return "SpatialRicciScalar"; }
  using type = Scalar<DataVector>;
};

template <typename Frame>
struct SpatialRicciScalarCompute : SpatialRicciScalar, db::ComputeTag {
  static constexpr auto function =
      &StrahlkorperGr::spatial_ricci_scalar<Frame>;
  using argument_tags =
      tmpl::list<gr::Tags::RicciTensor<3, Frame, DataVector>,
                 gr::Tags::InverseSpatialMetric<3, Frame, DataVector>>;
};

/// Computes the area element on a Strahlkorper. Useful for integrals.
template <typename Frame>
struct AreaElement : db::ComputeTag {
  static std::string name() noexcept { return "AreaElement"; }
  static constexpr auto function = area_element<Frame>;
  using argument_tags = tmpl::list<
      gr::Tags::SpatialMetric<3, Frame>, StrahlkorperTags::Jacobian<Frame>,
      StrahlkorperTags::NormalOneForm<Frame>, StrahlkorperTags::Radius<Frame>,
      StrahlkorperTags::Rhat<Frame>>;
};

/// Calculates the spin function, \f$\Omega\f$, a component of the spin
/// vector.

struct SpinFunction : db::SimpleTag {
  static std::string name() noexcept { return "SpinFunction"; }
  using type = Scalar<DataVector>;
};

template <typename Frame>
struct SpinFunctionCompute : SpinFunction, db::ComputeTag {
  static constexpr auto function = &StrahlkorperGr::spin_function<Frame>;
  using argument_tags =
      tmpl::list<StrahlkorperTags::Tangents<Frame>,
                 StrahlkorperTags::YlmSpherepack, UnitNormalVector<Frame>,
                 AreaElement<Frame>,
                 gr::Tags::ExtrinsicCurvature<3, Frame, DataVector>>;
};

/// Measures the quasilocal spin magnitude of a Strahlkorper, by
/// inserting \f$\alpha=1\f$ into Eq. (10) of \cite Owen2009sb
/// and dividing by \f$8\pi\f$ to yield the spin magnitude. The
/// spin magnitude is a Euclidean norm of surface integrals over the horizon
/// \f$S = \frac{1}{8\pi}\oint z \Omega dA\f$,
/// where \f$\Omega\f$ is obtained via `StrahlkorperGr::spin_function()`,
/// \f$dA\f$ is the area element, and \f$z\f$ (the "spin potential") is a
/// solution of a generalized eigenproblem given by Eq. (9) of
/// \cite Owen2009sb. Specifically,
/// \f$\nabla^4 z + \nabla \cdot (R\nabla z) = \lambda \nabla^2 z\f$, where
/// \f$R\f$ is obtained via `StrahlkorperGr::ricci_scalar()`. The spin
/// magnitude is the Euclidean norm of the three values of \f$S\f$ obtained from
/// the eigenvectors \f$z\f$ with the 3 smallest-magnitude
/// eigenvalues \f$\lambda\f$. Note that this formulation of the eigenproblem
/// uses the "Owen" normalization, Eq. (A9) and surrounding discussion in
/// \cite Lovelace2008tw.
/// The eigenvectors are normalized  with the "Kerr normalization",
/// Eq. (A22) of \cite Lovelace2008tw.
/// The argument `spatial_metric` is the metric of the 3D spatial slice
/// evaluated on the `Strahlkorper`.
/// The argument `tangents` can be obtained from the StrahlkorperDataBox
/// using the `StrahlkorperTags::Tangents` tag, and the argument
/// `unit_normal_vector` can
/// be found by raising the index of the one-form returned by
/// `StrahlkorperGr::unit_normal_one_form`.
/// The argument `ylm` is the `YlmSpherepack` of the `Strahlkorper`.
/// The argument `area_element`
/// can be computed via `StrahlkorperGr::area_element`.


struct DimensionfulSpinMagnitude : db::SimpleTag {
  static std::string name() noexcept { return "DimensionfulSpinMagnitude"; }
  using type = double;
};

template <typename Frame>
struct DimensionfulSpinMagnitudeCompute : DimensionfulSpinMagnitude,
                                          db::ComputeTag {
  static constexpr auto function =
      &StrahlkorperGr::dimensionful_spin_magnitude<Frame>;
  using argument_tags =
      tmpl::list<HorizonRicciScalar, SpinFunction,
                 gr::Tags::SpatialMetric<3, Frame>,
                 StrahlkorperTags::Tangents<Frame>,
                 StrahlkorperTags::YlmSpherepack, AreaElement<Frame>>;
};

struct DimensionfulSpinVector : db::SimpleTag {
  static std::string name() noexcept { return "DimensionfulSpinVector"; }
  using type = std::array<double, 3>;
};

template <typename Frame>
struct DimensionfulSpinVectorCompute : DimensionfulSpinVector, db::ComputeTag {
  static std::array<double, 3> function(
      const double& dimensionful_spin_magnitude,
      const Scalar<DataVector>& area_element, const DataVector& radius,
      const tnsr::i<DataVector, 3, Frame>& r_hat,
      const Scalar<DataVector>& ricci_scalar,
      const Scalar<DataVector>& spin_function,
      const YlmSpherepack& ylm) noexcept {
    return StrahlkorperGr::spin_vector<Frame>(
        dimensionful_spin_magnitude, area_element,
        Scalar<DataVector>{std::move(radius)}, r_hat, ricci_scalar,
        spin_function, ylm);
  }
  using argument_tags =
      tmpl::list<DimensionfulSpinMagnitude, AreaElement<Frame>,
                 StrahlkorperTags::Radius<Frame>, StrahlkorperTags::Rhat<Frame>,
                 HorizonRicciScalar, SpinFunction,
                 StrahlkorperTags::YlmSpherepack>;
};

/// Computes the integral of a scalar over a Strahlkorper.
template <typename IntegrandTag, typename Frame>
struct SurfaceIntegral : db::ComputeTag {
  static std::string name() noexcept {
    return "SurfaceIntegral" + IntegrandTag::name();
  }
  static constexpr auto function = surface_integral_of_scalar<Frame>;
  using argument_tags = tmpl::list<AreaElement<Frame>, IntegrandTag,
                                   StrahlkorperTags::Strahlkorper<Frame>>;
};

struct Unity : db::ComputeTag {
  static std::string name() noexcept { return "Unity"; }
  static Scalar<DataVector> function(
      const Scalar<DataVector>& used_for_size) noexcept {
    return make_with_value<Scalar<DataVector>>(used_for_size, 1.0);
  }
  using argument_tags = tmpl::list<AreaElement<Frame::Inertial>>;
};

/// Computes the area of a horizon as a double.

struct Area : db::SimpleTag {
  static std::string name() noexcept { return "Area"; }
  using type = double;
};

template <typename Frame>
struct AreaCompute : Area, db::ComputeTag {
  static double function(const Strahlkorper<Frame>& strahlkorper,
                         const Scalar<DataVector>& area_element) noexcept {
    return strahlkorper.ylm_spherepack().definite_integral(
        get(area_element).data());
  }
  using argument_tags =
      tmpl::list<StrahlkorperTags::Strahlkorper<Frame>, AreaElement<Frame>>;
};

/// See Eqs. (15.38) \cite Hartle2003gravity. This function computes the
/// irreducible mass from the area of a horizon. Specifically, computes
/// \f$M_\mathrm{irr}=\sqrt{\frac{A}{16\pi}}\f$.

struct IrreducibleMass : db::SimpleTag {
  static std::string name() noexcept { return "IrreducibleMass"; }
  using type = double;
};

template <typename Frame>
struct IrreducibleMassCompute : IrreducibleMass, db::ComputeTag {
  static constexpr auto function = &irreducible_mass;
  using argument_tags = tmpl::list<AreaCompute<Frame>>;
};

/// See e.g. Eq. (1) of \cite Lovelace2016uwp. This function computes the
/// Christodoulou mass from the dimensionful spin angular momentum \f$S\f$
/// and the irreducible mass \f$M_{irr}\f$ of a black hole horizon.
/// Specifically, computes \f$M=\sqrt{M_{irr}^2+\frac{S^2}{4M_{irr}^2}}\f$.

struct ChristodoulouMass : db::SimpleTag {
  static std::string name() noexcept { return "ChristodoulouMass"; }
  using type = double;
};

struct ChristodoulouMassCompute : ChristodoulouMass, db::ComputeTag {
  static constexpr auto function = &StrahlkorperGr::christodoulou_mass;
  using argument_tags =
      tmpl::list<DimensionfulSpinMagnitude, IrreducibleMass>;
};
}  // namespace Tags
}  // namespace StrahlkorperGr
