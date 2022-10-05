// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Executables/GeneralizedHarmonic/EvolveGhBinaryBlackHole.hpp"
#include "Options/ParseOptions.tpp"
#include "Parallel/MainOptionList.hpp"
#include "Utilities/TaggedTuple.hpp"

template void Options::create_all_options<
    typename Parallel::get_main_option_list<EvolutionMetavars>::type,
    EvolutionMetavars>();
