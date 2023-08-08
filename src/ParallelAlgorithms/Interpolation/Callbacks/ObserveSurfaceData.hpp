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
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace intrp {
namespace callbacks {

/// \brief post_interpolation_callback that outputs
/// 2D "volume" data on a surface.
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
    const observers::ObservationId& observation_id = observers::ObservationId(
        InterpolationTarget_detail::get_temporal_id_value(temporal_id),
        subfile_path + ".vol");

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

    const size_t l_max = ylm.l_max();
    // l_max == m_max
    const size_t num_coefficients =
        ylm::Spherepack::physical_size(l_max, l_max);

    const std::array<double, 3> expansion_center =
        strahlkorper.expansion_center();

    // l_max + 3 dims of expansion center = 4 columns
    const size_t num_columns = num_coefficients + 4;

    std::vector<std::string> ylm_legend;
    ylm_legend.reserve(num_columns);
    std::vector<double> ylm_data;
    ylm_data.reserve(num_columns);
    // ylm_legend[0] = "Lmax";
    // ylm_legend[1] = "Mmax";
    // ylm_legend[2] = "ExpansionCenter_x";
    // ylm_legend[3] = "ExpansionCenter_y";
    // ylm_legend[4] = "ExpansionCenter_z";
    ylm_legend.emplace_back("ExpansionCenter_x");
    ylm_data.emplace_back(expansion_center[0]);
    ylm_legend.emplace_back("ExpansionCenter_y");
    ylm_data.emplace_back(expansion_center[1]);
    ylm_legend.emplace_back("ExpansionCenter_z");
    ylm_data.emplace_back(expansion_center[2]);
    ylm_legend.emplace_back("Lmax");
    ylm_data.emplace_back(l_max);

    const DataVector ylm_coefficients = strahlkorper.coefficients();
    // l_max == m_max
    SpherepackIterator iter(l_max, l_max);
    for (size_t l = 0; l <= l_max; l++) {
      for (int m = -l; m <= static_cast<int>(l); m++) {
        ylm_legend.push_back(MakeString{} << "coef(" << l << "," << m << ")");

        iter.set(l, m);
        ylm_data.push_back(ylm_coefficients[iter()]);
      }
    }

    const std::string ylm_subfile_name{std::string{"/"} + surface_name +
                                       "_Ylm"};

    ASSERT(ylm_legend.size() == ylm_data.size(),
           "Legend (" << ylm_legend.size()
                      << ") does not have the same number of "
                         "components as data to write ("
                      << ylm_data.size() << ")");

    // // const auto& reduction_file_lock =
    // //     db::get<observers::Tags::H5FileLock>(
    // //         make_not_null(&box));
    // // const std::lock_guard hold_lock(reduction_file_lock);
    // observers::ThreadedActions::ReductionActions_detail::write_data(
    //     ylm_subfile_name, observers::input_source_from_cache(cache),
    //     ylm_legend, std::make_tuple(ylm_data),
    //     Parallel::get<observers::Tags::ReductionFileName>(cache),
    //     std::index_sequence<0>{});
    Parallel::threaded_action<
        observers::ThreadedActions::WriteReductionDataRow>(
        // Node 0 is always the writer, so directly call the component on that
        // node
        proxy[0], ylm_subfile_name, ylm_legend, std::make_tuple(ylm_data));
  }
};
}  // namespace callbacks
}  // namespace intrp
