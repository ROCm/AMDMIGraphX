//      
//     Supporting functions for enum values used in operator parameters.
//     These values are declared as "enum class" and should include << streaming operators
//     to be able to write their values in human-readable format so users can
//     save and edit model files.
//
#include <sstream>
#include <migraphx/op/common.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

std::ostream& operator<<(std::ostream& os, pooling_mode v)
{
    // the strings for the enum are the same as the values used for onnx parsing
    // but this enum is not onnx-specific:  strings must be converted when parsing tf
    static const std::vector<std::string> pooling_mode_str = {"average", "max"};
    os << pooling_mode_str[static_cast<std::underlying_type<pooling_mode>::type>(v)];
    return os;
}
std::ostream& operator<<(std::ostream& os, rnn_direction v)
{
    static const std::vector<std::string> rnn_direction_str = {
        "forward", "reverse", "bidirectional"};
    os << rnn_direction_str[static_cast<std::underlying_type<rnn_direction>::type>(v)];
    return os;
}

// roi_align roialign_mode
std::ostream& operator<<(std::ostream& os, roialign_mode v)
{
    static const std::vector<std::string> roi_str = {"average", "max"};
    os << roi_str[static_cast<std::underlying_type<roialign_mode>::type>(v)];
    return os;
}

std::string to_str(pooling_mode v)
{
    std::stringstream ss("");
    ss << v;
    return ss.str();
}

std::string to_str(roialign_mode v)
{
    std::stringstream ss("");
    ss << v;
    return ss.str();
}

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
