// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Parallel/PhaseControl/CheckpointAndExitAfterWallclock.hpp"

#include <optional>
#include <pup.h>
#include <sstream>

#include "Options/Options.hpp"
#include "Parallel/Phase.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace PhaseControl {

namespace Tags {
std::optional<Parallel::Phase> RestartPhase::combine_method::operator()(
    const std::optional<Parallel::Phase> /*first_phase*/,
    const std::optional<Parallel::Phase>& /*second_phase*/) {
  std::ostringstream ss;
  ss << "The restart phase should only be altered by the phase change "
        "arbitration in the Main chare, so no reduction data should be "
        "provided.";
  ERROR(ss);
}

std::optional<double> WallclockHoursAtCheckpoint::combine_method::operator()(
    const std::optional<double> /*first_time*/,
    const std::optional<double>& /*second_time*/) {
  std::ostringstream ss;
  ss << "Must be in the Initialization phase.";
  ERROR(ss);
  ERROR(
      "The wallclock time at which a checkpoint was requested should "
      "only be altered by the phase change arbitration in the Main "
      "chare, so no reduction data should be provided.");
}
}  // namespace Tags

CheckpointAndExitAfterWallclock::CheckpointAndExitAfterWallclock(
    const std::optional<double> wallclock_hours,
    const Options::Context& context)
    : wallclock_hours_for_checkpoint_and_exit_(wallclock_hours) {
  if (wallclock_hours.has_value() and wallclock_hours.value() < 0.0) {
    std::ostringstream ss;
    ss << "Must give a positive time in hours, but got "
       << wallclock_hours.value();
    PARSE_ERROR(context, ss);
  }
}

CheckpointAndExitAfterWallclock::CheckpointAndExitAfterWallclock(
    CkMigrateMessage* msg)
    : PhaseChange(msg) {}

void CheckpointAndExitAfterWallclock::pup(PUP::er& p) {
  PhaseChange::pup(p);
  p | wallclock_hours_for_checkpoint_and_exit_;
}
}  // namespace PhaseControl

PUP::able::PUP_ID PhaseControl::CheckpointAndExitAfterWallclock::my_PUP_ID = 0;
