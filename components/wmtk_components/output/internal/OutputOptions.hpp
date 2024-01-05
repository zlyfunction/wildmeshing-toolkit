#pragma once

#include <wmtk_components/base/json_utils.hpp>

#include <nlohmann/json.hpp>

namespace wmtk {
namespace components {
namespace internal {

struct OutputOptions
{
    std::string type;
    std::string input;
    std::filesystem::path file;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(OutputOptions, type, input, file);

} // namespace internal
} // namespace components
} // namespace wmtk