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

template <typename Frame>
struct MaxRadius : db::ComputeTag {
  static std::string name() noexcept { return "MaxRadius"; }
  static double function(
      const DataVector& radius) noexcept {
    return max(radius);
  }
  using argument_tags = tmpl::list<Radius<Frame>>;
};

template <typename Frame>
struct MinRadius : db::ComputeTag {
  static std::string name() noexcept { return "MinRadius"; }
  static double function(
      const DataVector& radius) noexcept {
    return min(radius);
  }
  using argument_tags = tmpl::list<Radius<Frame>>;
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
                 NormalOneForm<Frame>>;
};

template <typename Frame>
struct UnitNormalOneForm : db::SimpleTag {
  static std::string name() noexcept { return "UnitNormalOneForm"; }
  using type = tnsr::i<DataVector, 3, Frame>;
};

template <typename Frame>
struct UnitNormalOneFormCompute : UnitNormalOneForm<Frame>, db::ComputeTag {
  static constexpr auto function = &StrahlkorperGr::unit_normal_one_form<Frame>;
  using argument_tags =
      tmpl::list<NormalOneForm<Frame>, OneOverOneFormMagnitude>;
};

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

struct RicciScalar : db::SimpleTag {
  static std::string name() noexcept { return "RicciScalar"; }
  using type = Scalar<DataVector>;
};

template <typename Frame>
struct RicciScalarCompute : RicciScalar, db::ComputeTag {
  static constexpr auto function = &StrahlkorperGr::ricci_scalar<Frame>;
  using argument_tags =
      tmpl::list<gr::Tags::RicciTensor<3, Frame, DataVector>,
                 StrahlkorperTags::UnitNormalVector<Frame>,
                 gr::Tags::ExtrinsicCurvature<3, Frame, DataVector>,
                 gr::Tags::InverseSpatialMetric<3, Frame, DataVector>>;
};

struct MaxRicciScalar : db::SimpleTag {
  static std::string name() noexcept { return "MaxRicciScalar"; }
  using type = double;
};

struct MaxRicciScalarCompute : MaxRicciScalar, db::ComputeTag {
  static double function(const Scalar<DataVector>& ricci_scalar) {
    return max(get(ricci_scalar));
  }
  using argument_tags = tmpl::list<RicciScalar>;
};

struct MinRicciScalar : db::SimpleTag {
  static std::string name() noexcept { return "MinRicciScalar"; }
  using type = double;
};

struct MinRicciScalarCompute : MinRicciScalar, db::ComputeTag {
  static double function(const Scalar<DataVector>& ricci_scalar) {
    return min(get(ricci_scalar));
  }
  using argument_tags = tmpl::list<RicciScalar>;
};

struct RicciScalar3D : db::SimpleTag {
  static std::string name() noexcept { return "RicciScalar"; }
  using type = Scalar<DataVector>;
};

template <typename Frame>
struct RicciScalar3DCompute : RicciScalar3D, db::ComputeTag {
  static constexpr auto function = &StrahlkorperGr::ricci_scalar_3d<Frame>;
  using argument_tags =
      tmpl::list<gr::Tags::RicciTensor<3, Frame, DataVector>,
                 gr::Tags::InverseSpatialMetric<3, Frame, DataVector>>;
};

struct MaxRicciScalar3D : db::SimpleTag {
  static std::string name() noexcept { return "MaxRicciScalar3D"; }
  using type = double;
};

struct MaxRicciScalar3DCompute : MaxRicciScalar3D, db::ComputeTag {
  static double function(const Scalar<DataVector>& ricci_scalar) {
    return max(get(ricci_scalar));
  }
  using argument_tags = tmpl::list<RicciScalar3D>;
};

struct MinRicciScalar3D : db::SimpleTag {
  static std::string name() noexcept { return "MinRicciScalar3D"; }
  using type = double;
};

struct MinRicciScalar3DCompute : MinRicciScalar3D, db::ComputeTag {
  static double function(const Scalar<DataVector>& ricci_scalar) {
    return min(get(ricci_scalar));
  }
  using argument_tags = tmpl::list<RicciScalar3D>;
};

struct RicciScalar2ndTerm : db::SimpleTag {
  static std::string name() noexcept { return "RicciScalar2ndTerm"; }
  using type = Scalar<DataVector>;
};

template <typename Frame>
struct RicciScalar2ndTermCompute : RicciScalar2ndTerm, db::ComputeTag {
  static Scalar<DataVector> function(
      const tnsr::ii<DataVector, 3, Frame>& spatial_ricci_tensor,
      const tnsr::I<DataVector, 3, Frame>& unit_normal_vector,
      const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
      const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric) noexcept {
    auto ricci_scalar = trace(spatial_ricci_tensor, inverse_spatial_metric);

    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        get(ricci_scalar) -= 2.0 * spatial_ricci_tensor.get(i, j) *
                             unit_normal_vector.get(i) *
                             unit_normal_vector.get(j);

        for (size_t k = 0; k < 3; ++k) {
          for (size_t l = 0; l < 3; ++l) {
            // K^{ij} K_{ij} = g^{ik} g^{jl} K_{kl} K_{ij}
            get(ricci_scalar) -=
                inverse_spatial_metric.get(i, k) *
                inverse_spatial_metric.get(j, l) *
                extrinsic_curvature.get(k, l) *
                extrinsic_curvature.get(i, j);
          }
        }
      }
    }

    return ricci_scalar;
  }
  using argument_tags =
      tmpl::list<gr::Tags::RicciTensor<3, Frame, DataVector>,
                 StrahlkorperTags::UnitNormalVector<Frame>,
                 gr::Tags::ExtrinsicCurvature<3, Frame, DataVector>,
                 gr::Tags::InverseSpatialMetric<3, Frame, DataVector>>;
};

struct MaxRicciScalar2ndTerm : db::SimpleTag {
  static std::string name() noexcept { return "MaxRicciScalar2ndTerm"; }
  using type = double;
};

struct MaxRicciScalar2ndTermCompute : MaxRicciScalar2ndTerm, db::ComputeTag {
  static double function(const Scalar<DataVector>& ricci_scalar_second_term) {
    return max(get(ricci_scalar_second_term));
  }
  using argument_tags = tmpl::list<RicciScalar2ndTerm>;
};

struct MinRicciScalar2ndTerm : db::SimpleTag {
  static std::string name() noexcept { return "MinRicciScalar2ndTerm"; }
  using type = double;
};

struct MinRicciScalar2ndTermCompute : MinRicciScalar2ndTerm, db::ComputeTag {
  static double function(const Scalar<DataVector>& ricci_scalar_second_term) {
    return min(get(ricci_scalar_second_term));
  }
  using argument_tags = tmpl::list<RicciScalar2ndTerm>;
};

struct SpinFunction : db::SimpleTag {
  static std::string name() noexcept { return "SpinFunction"; }
  using type = Scalar<DataVector>;
};

template <typename Frame>
struct SpinFunctionCompute : SpinFunction, db::ComputeTag {
  static constexpr auto function = &StrahlkorperGr::spin_function<Frame>;
  using argument_tags =
      tmpl::list<StrahlkorperTags::Tangents<Frame>,
      StrahlkorperTags::YlmSpherepack,
      StrahlkorperTags::UnitNormalVector<Frame>,
      AreaElement<Frame>,
      gr::Tags::ExtrinsicCurvature<3, Frame, DataVector>>;
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

struct IrreducibleMass : db::SimpleTag {
  static std::string name() noexcept { return "IrreducibleMass"; }
  using type = double;
};

template <typename Frame>
struct IrreducibleMassCompute : IrreducibleMass, db::ComputeTag {
  static constexpr auto function = &irreducible_mass;
  using argument_tags = tmpl::list<AreaCompute<Frame>>;
};

}  // namespace Tags
}  // namespace StrahlkorperGr
