// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"

class DataVector;

namespace Ccz4 {
/// \brief Tags for the CCZ4 formulation of Einstein equations
namespace Tags {
template <typename DataType = DataVector>
struct ConformalFactor;
template <typename DataType = DataVector>
struct ConformalFactorSquared;
template <typename DataType = DataVector>
struct DetConformalSpatialMetric;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct ATilde;
template <typename DataType = DataVector>
struct TraceATilde;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct InverseATilde;
template <typename DataType = DataVector>
struct SlicingCondition;
template <typename DataType = DataVector>
struct DerivSlicingCondition;
template <typename DataType = DataVector>
struct Theta;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct b;
template <typename DataType = DataVector>
struct Eta;
template <typename DataType = DataVector>
struct LapseInitialTimeDerivative;
template <typename DataType = DataVector>
struct LogLapse;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct FieldA;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct FieldB;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct FieldD;
template <typename DataType = DataVector>
struct LogConformalFactor;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct FieldP;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct FieldDUp;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct ConformalChristoffelSecondKind;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct DerivConformalChristoffelSecondKind;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct ChristoffelSecondKind;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct Ricci;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct GradGradLapse;
template <typename DataType = DataVector>
struct DivergenceLapse;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct ContractedConformalChristoffelSecondKind;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct DerivContractedConformalChristoffelSecondKind;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct GammaHat;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct SpatialZ4Constraint;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct SpatialZ4ConstraintUp;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct GradSpatialZ4Constraint;
template <typename DataType = DataVector>
struct RicciScalarPlusDivergenceZ4Constraint;
}  // namespace Tags

/// \brief Input option tags for the CCZ4 evolution system
namespace OptionTags {
struct Group;
}  // namespace OptionTags
}  // namespace Ccz4
