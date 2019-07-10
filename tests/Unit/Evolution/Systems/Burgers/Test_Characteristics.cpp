// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Burgers/Characteristics.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Utilities/TMPL.hpp"

SPECTRE_TEST_CASE("Unit.Burgers.Characteristics", "[Unit][Burgers]") {
  const auto box = db::create<
      db::AddSimpleTags<Burgers::Tags::U>,
      db::AddComputeTags<Burgers::Tags::CharacteristicSpeedsCompute>>(
      Scalar<DataVector>{4.0});
  CHECK(db::get<Burgers::Tags::CharacteristicSpeedsCompute>(box)[0] == 4.0);
}
