#include <migraphx/onnx/map_activation_functions.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

const std::unordered_map<std::string, operation>& map_activation_functions()
{
    static const std::unordered_map<std::string, operation> m = {
        {"tanh", make_op("tanh")},
        {"relu", make_op("relu")},
        {"sigmoid", make_op("sigmoid")},
        {"leakyrelu", make_op("leaky_relu")},
        {"elu", make_op("elu")}};
    return m;
}

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
