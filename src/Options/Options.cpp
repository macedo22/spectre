// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <string>
#include <utility>
#include <yaml-cpp/yaml.h>

#include "Options/Options.hpp"
#include "Options/OptionsDetails.hpp"

namespace Options {
// clang-tidy: YAML::Node not movable (as of yaml-cpp-0.5.3)
// NOLINTNEXTLINE(performance-unnecessary-value-param)
Option::Option(YAML::Node node, Context context)
    : node_(std::make_unique<YAML::Node>(std::move(node))),
      context_(std::move(context)) {  // NOLINT
  context_.line = node.Mark().line;
  context_.column = node.Mark().column;
}

Option::Option(Context context)
    : node_(std::make_unique<YAML::Node>()), context_(std::move(context)) {}

const YAML::Node& Option::node() const { return *node_; }
const Context& Option::context() const { return context_; }

// Append a line to the contained context.
void Option::append_context(const std::string& context) {
  context_.append(context);
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
void Option::set_node(YAML::Node node) {
  // clang-tidy: YAML::Node not movable (as of yaml-cpp-0.5.3)
  *node_ = std::move(node);  // NOLINT
  context_.line = node_->Mark().line;
  context_.column = node_->Mark().column;
}
}  // namespace Options
