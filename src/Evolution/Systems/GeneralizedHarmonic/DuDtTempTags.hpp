// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
/// \endcond

namespace gh {
namespace Tags {
/// \f$\gamma_1 \gamma_2\f$ constraint damping product
struct Gamma1Gamma2 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// \f$0.5\Pi_{ab}n^an^b\f$
struct HalfPiTwoNormals : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// \f$n^a \mathcal{C}_a\f$
struct NormalDotOneIndexConstraint : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// \f$\gamma_1 + 1\f$
struct Gamma1Plus1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// \f$\Pi_{ab}n^a\f$
template <size_t Dim>
struct PiOneNormal : db::SimpleTag {
  using type = tnsr::a<DataVector, Dim, Frame::Inertial>;
};

/// \f$0.5\Phi_{iab}n^an^b\f$
template <size_t Dim>
struct HalfPhiTwoNormals : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};

/// \f$\beta^i \mathcal{C}_{iab}\f$
template <size_t Dim>
struct ShiftDotThreeIndexConstraint : db::SimpleTag {
  using type = tnsr::aa<DataVector, Dim, Frame::Inertial>;
};

/// \f$\v^i_g \mathcal{C}_{iab}\f$
template <size_t Dim>
struct MeshVelocityDotThreeIndexConstraint : db::SimpleTag {
  using type = tnsr::aa<DataVector, Dim, Frame::Inertial>;
};

/// \f$\Phi_{iab}n^a\f$
template <size_t Dim>
struct PhiOneNormal : db::SimpleTag {
  using type = tnsr::ia<DataVector, Dim, Frame::Inertial>;
};

/// \f$\Pi_a{}^b\f$
template <size_t Dim>
struct PiSecondIndexUp : db::SimpleTag {
  using type = tnsr::aB<DataVector, Dim, Frame::Inertial>;
};

/// \f$\Phi^i{}_{ab}\f$
template <size_t Dim>
struct PhiFirstIndexUp : db::SimpleTag {
  using type = tnsr::Iaa<DataVector, Dim, Frame::Inertial>;
};

/// \f$\Phi_{ia}{}^b\f$
template <size_t Dim>
struct PhiThirdIndexUp : db::SimpleTag {
  using type = tnsr::iaB<DataVector, Dim, Frame::Inertial>;
};

/// \f$\Gamma_{ab}{}^c\f$
template <size_t Dim>
struct SpacetimeChristoffelFirstKindThirdIndexUp : db::SimpleTag {
  using type = tnsr::abC<DataVector, Dim, Frame::Inertial>;
};

template <size_t Dim>
struct InverseSpatialMetricLogical1 : db::SimpleTag {
  using type =
      Tensor<DataVector, Symmetry<2, 1>,
             index_list<SpatialIndex<Dim, UpLo::Up, Frame::ElementLogical>,
                        SpatialIndex<Dim, UpLo::Up, Frame::Inertial>>>;
};

template <size_t Dim>
struct ShiftDotDSpacetimeMetric : db::SimpleTag {
  using type = tnsr::aa<DataVector, Dim>;
};

template <size_t Dim>
struct ShiftDotPhi : db::SimpleTag {
  using type = tnsr::aa<DataVector, Dim>;
};

template <size_t Dim>
struct MeshVelocityDotPhi : db::SimpleTag {
  using type = tnsr::aa<DataVector, Dim>;
};

template <size_t Dim>
struct MeshVelocityDotDSpacetimeMetric : db::SimpleTag {
  using type = tnsr::aa<DataVector, Dim>;
};

/*!
 * \brief The inverse gauge source function for the generalized harmonic system.
 *
 * \details Defined as \f$ H^b = g^{ab} H_a\f$ where \f$ H_a\f$ is defined by
 * `GaugeH`.
 */
template <size_t Dim>
struct UpperGaugeH : db::SimpleTag {
  using type = tnsr::A<DataVector, Dim, Frame::Inertial>;
};

template <size_t Dim>
struct Gamma2LogicalDSpacetimeMetricMinusLogicalDPi : db::SimpleTag {
  using type =
      Tensor<DataVector, Symmetry<2, 1, 1>,
             index_list<SpatialIndex<Dim, UpLo::Lo, Frame::ElementLogical>,
                        SpacetimeIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Lo, Frame::Inertial>>>;
};

}  // namespace Tags
}  // namespace gh
