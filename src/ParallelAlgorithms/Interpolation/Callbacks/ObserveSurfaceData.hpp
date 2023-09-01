// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/PostInterpolationCallback.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace intrp {
namespace callbacks {
namespace detail {
/// \brief Fills the legend and row of spherical harmonic data to write to disk
///
/// \details
/// The number of coefficients to write is based on `max_l`, the maximum value
/// that the input `strahlkorper` could possibly have. When
/// `strahlkorper.l_max() < max_l`, coefficients with \f$l\f$ higher than
/// `strahlkorper.l_max()` will simply be zero. Assuming the same `max_l` is
/// always used for a given surface, we will always write the same number of
/// columns for each row.
///
/// \param legend the headers for the strahlkorper data
/// \param data the spherical harmonic data to write
/// \param strahlkorper the strahlkorper from which to get the spherical
/// harmonic data to be written
/// \param time the current time
/// \param max_l the maximum value that \f$l_max\f$ could be (sets number of
/// columns to write)
template <typename Frame>
void fill_ylm_legend_and_data(
    const gsl::not_null<std::vector<std::string>*> legend,
    const gsl::not_null<std::vector<double>*> data,
    const Strahlkorper<Frame>& strahlkorper, const double time,
    const size_t max_l) {
  const size_t l_max = strahlkorper.l_max();
  ASSERT(max_l >= l_max,
         "The Lmax of the Ylm data to write ("
             << l_max << ") is larger than the maximum value that l can be ("
             << max_l << ").");
  const std::array<double, 3> expansion_center =
      strahlkorper.expansion_center();
  // number of terms in
  // \sum_{l=0}^{max\_l} \sum_{m=-l}^{l} F^{lm} Y^{lm}(\theta,\phi) is the
  // sum of the first (max_l + 1) odd numbers, which is (max_l + 1)^2
  const size_t num_coefficients = square(max_l + 1);
  // time + 3 dims of expansion center + Lmax = 5 columns
  const size_t num_columns = num_coefficients + 5;

  legend->reserve(num_columns);
  data->reserve(num_columns);

  legend->emplace_back("Time");
  data->emplace_back(time);
  legend->emplace_back("ExpansionCenter_x");
  data->emplace_back(expansion_center[0]);
  legend->emplace_back("ExpansionCenter_y");
  data->emplace_back(expansion_center[1]);
  legend->emplace_back("ExpansionCenter_z");
  data->emplace_back(expansion_center[2]);
  legend->emplace_back("Lmax");
  data->emplace_back(l_max);

  const DataVector ylm_coefficients = strahlkorper.coefficients();
  // fill coefficients for l in [0, l_max]
  // l_max == m_max
  SpherepackIterator iter(l_max, l_max);
  for (size_t l = 0; l <= l_max; l++) {
    for (int m = -l; m < 0; m++) {
      legend->push_back(MakeString{} << "Im(" << l << "," << m << ")");

      iter.set(l, m);
      data->push_back(ylm_coefficients[iter()]);
    }
    for (int m = 0; m <= static_cast<int>(l); m++) {
      legend->push_back(MakeString{} << "Re(" << l << "," << m << ")");

      iter.set(l, m);
      data->push_back(ylm_coefficients[iter()]);
    }
  }
  // fill coefficients for l in [l_max + 1, max_l],
  // i.e. higher order coefficients beyond this Strahlkorper's l_max == 0.0
  data->resize(num_columns, 0.0);
  for (size_t l = l_max + 1; l <= max_l; l++) {
    for (int m = -l; m < 0; m++) {
      legend->push_back(MakeString{} << "Im(" << l << "," << m << ")");
    }
    for (int m = 0; m <= static_cast<int>(l); m++) {
      legend->push_back(MakeString{} << "Re(" << l << "," << m << ")");
    }
  }

  ASSERT(legend->size() == data->size(),
         "Size of Ylm legend ("
             << legend->size()
             << ") does not have the same number of columns as data to write ("
             << data->size() << ")");
}
}  // namespace detail

/// \brief post_interpolation_callback that outputs 2D "volume" data on a
/// surface and the surface's spherical harmonic data
///
/// Uses:
/// - Metavariables
///   - `temporal_id`
/// - DataBox:
///   - `TagsToObserve` (each tag must be a Scalar<DataVector>)
///
/// Conforms to the intrp::protocols::PostInterpolationCallback protocol
///
/// For requirements on InterpolationTargetTag, see
/// intrp::protocols::InterpolationTargetTag
///
/// \details The columns of spherical harmonic data written take the form
/// `[Time, ExpansionCenter_x, ExpansionCenter_y, Expansion_Center_z, Lmax,
/// Re(0,0), Im(1,-1), Re(1,0), Re(1,1), Im(2,-2), Im(2,-1), Re(2,0), Re(2,1),
/// Re(2,2), ..., Re(Lmax,Lmax)]`, where the number of coefficients is equal to
/// \f$(Lmax + 1)^2\f$ and the coefficients written are the strahlkorper
/// coefficients stored and defined by `::Strahlkorper::coefficients()`.
///
/// \note Currently, \f$l_{max}\f$ for a given surface does not change over the
/// course of the simulation, which means that the total number of columns of
/// coefficients that we need to write is also constant. The current
/// implementation of writing the coefficients at one time assumes \f$l_{max}\f$
/// of a surface remains constant. If and when in the future functionality for
/// an adaptive \f$l_{max}\f$ is added, the implementation for writing the
/// coefficients will need to be updated to account for this. One possible way
/// to address this is to have a known maximum \f$l_{max}\f$ for a given surface
/// and write all coefficients up to that maximum \f$l_{max}\f$.
template <typename TagsToObserve, typename InterpolationTargetTag,
          typename HorizonFrame>
struct ObserveSurfaceData
    : tt::ConformsTo<intrp::protocols::PostInterpolationCallback> {
  static constexpr double fill_invalid_points_with =
      std::numeric_limits<double>::quiet_NaN();

  using const_global_cache_tags = tmpl::list<observers::Tags::SurfaceFileName>;

  template <typename DbTags, typename Metavariables, typename TemporalId>
  static void apply(const db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const TemporalId& temporal_id) {
    const Strahlkorper<HorizonFrame>& strahlkorper =
        get<StrahlkorperTags::Strahlkorper<HorizonFrame>>(box);
    const ylm::Spherepack& ylm = strahlkorper.ylm_spherepack();

    // Output the inertial-frame coordinates of the Stralhlkorper.
    // Note that these coordinates are not
    // Spherepack-evenly-distributed over the inertial-frame sphere
    // (they are Spherepack-evenly-distributed over the HorizonFrame
    // sphere).
    std::vector<TensorComponent> tensor_components;
    if constexpr (db::tag_is_retrievable_v<
                      StrahlkorperTags::CartesianCoords<::Frame::Inertial>,
                      db::DataBox<DbTags>>) {
      const auto& inertial_strahlkorper_coords =
          get<StrahlkorperTags::CartesianCoords<::Frame::Inertial>>(box);
      tensor_components.push_back(
          {"InertialCoordinates_x"s, get<0>(inertial_strahlkorper_coords)});
      tensor_components.push_back(
          {"InertialCoordinates_y"s, get<1>(inertial_strahlkorper_coords)});
      tensor_components.push_back(
          {"InertialCoordinates_z"s, get<2>(inertial_strahlkorper_coords)});
    }

    // Output each tag if it is a scalar. Otherwise, throw a compile-time
    // error. This could be generalized to handle tensors of nonzero rank by
    // looping over the components, so each component could be visualized
    // separately as a scalar. But in practice, this generalization is
    // probably unnecessary, because Strahlkorpers are typically only
    // visualized with scalar quantities (used set the color at different
    // points on the surface).
    tmpl::for_each<TagsToObserve>([&box, &tensor_components](auto tag_v) {
      using Tag = tmpl::type_from<decltype(tag_v)>;
      const auto tag_name = db::tag_name<Tag>();
      const auto& tensor = get<Tag>(box);
      for (size_t i = 0; i < tensor.size(); ++i) {
        tensor_components.emplace_back(tag_name + tensor.component_suffix(i),
                                       tensor[i]);
      }
    });

    const std::string& surface_name =
        pretty_type::name<InterpolationTargetTag>();
    const std::string subfile_path{std::string{"/"} + surface_name};
    const std::vector<size_t> extents_vector{
        {ylm.physical_extents()[0], ylm.physical_extents()[1]}};
    const std::vector<Spectral::Basis> bases_vector{
        2, Spectral::Basis::SphericalHarmonic};
    const std::vector<Spectral::Quadrature> quadratures_vector{
        {Spectral::Quadrature::Gauss, Spectral::Quadrature::Equiangular}};
    const double time =
        InterpolationTarget_detail::get_temporal_id_value(temporal_id);
    const observers::ObservationId& observation_id =
        observers::ObservationId(time, subfile_path + ".vol");

    auto& proxy = Parallel::get_parallel_component<
        observers::ObserverWriter<Metavariables>>(cache);

    // We call this on proxy[0] because the 0th element of a NodeGroup is
    // always guaranteed to be present.
    Parallel::threaded_action<observers::ThreadedActions::WriteVolumeData>(
        proxy[0], Parallel::get<observers::Tags::SurfaceFileName>(cache),
        subfile_path, observation_id,
        std::vector<ElementVolumeData>{{surface_name, tensor_components,
                                        extents_vector, bases_vector,
                                        quadratures_vector}});

    std::vector<std::string> ylm_legend;
    std::vector<double> ylm_data;
    // The number of coefficients written will be (l_max + 1)^2 where l_max is
    // the current value of l_max for this surface's strahlkorper. Because l_max
    // remains constant, the number of coefficient columns written does, too. In
    // the future when l_max is adaptive, instead of passing in the current
    // l_max of the strahlkorper, we could pass in the maximum value that l_max
    // could be to ensure that we (a) have enough columns to write all the
    // coefficients regardless of the current value of l_max and (b) write a
    // constant number of columns for each row of data regardless of the current
    // l_max.
    detail::fill_ylm_legend_and_data(make_not_null(&ylm_legend),
                                     make_not_null(&ylm_data), strahlkorper,
                                     time, strahlkorper.l_max());

    const std::string ylm_subfile_name{std::string{"/"} + surface_name +
                                       "_Ylm"};

    Parallel::threaded_action<
        observers::ThreadedActions::WriteReductionDataRow>(
        proxy[0], ylm_subfile_name, ylm_legend, std::make_tuple(ylm_data));
  }
};
}  // namespace callbacks
}  // namespace intrp
