# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AddSubtract.hpp
  ConcreteTimeIndex.hpp
  Contract.hpp
  Evaluate.hpp
  LhsTensorSymmAndIndices.hpp
  Product.hpp
  NumberAsExpression.hpp
  SpatialSpacetimeIndex.hpp
  SquareRoot.hpp
  SymmetryTransformation.hpp
  TensorAsExpression.hpp
  TensorExpression.hpp
  TensorIndex.hpp
  TensorIndexTransformation.hpp
  )
