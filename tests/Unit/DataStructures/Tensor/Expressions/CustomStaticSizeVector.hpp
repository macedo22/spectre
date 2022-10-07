// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>

#include "DataStructures/VectorImpl.hpp"

// A VectorImpl whose static size limit is different than the default value
class CustomStaticSizeVector;
namespace blaze {
DECLARE_GENERAL_VECTOR_BLAZE_TRAITS(CustomStaticSizeVector);
}  // namespace blaze
class CustomStaticSizeVector
    : public VectorImpl<std::complex<double>, CustomStaticSizeVector,
                        (default_vector_impl_static_size + 1)> {
 public:
  CustomStaticSizeVector() = default;
  CustomStaticSizeVector(const CustomStaticSizeVector&) = default;
  CustomStaticSizeVector(CustomStaticSizeVector&&) = default;
  CustomStaticSizeVector& operator=(const CustomStaticSizeVector&) = default;
  CustomStaticSizeVector& operator=(CustomStaticSizeVector&&) = default;
  ~CustomStaticSizeVector() = default;

  static constexpr size_t static_size = default_vector_impl_static_size + 1;

  using BaseType =
      VectorImpl<std::complex<double>, CustomStaticSizeVector, static_size>;

  using BaseType::operator=;
  using BaseType::VectorImpl;
};
