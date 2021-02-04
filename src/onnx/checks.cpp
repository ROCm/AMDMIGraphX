#include <migraphx/onnx/checks.hpp>
#include <migraphx/errors.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

void check_arg_empty(const argument& arg, const std::string& msg)
{
    if(arg.empty())
    {
        MIGRAPHX_THROW(msg);
    }
}

void check_attr_sizes(size_t kdims, size_t attr_size, const std::string& error_msg)
{
    if(kdims != attr_size)
    {
        MIGRAPHX_THROW(error_msg + " k-dims: " + std::to_string(kdims) +
                       " attribute size: " + std::to_string(attr_size));
    }
}

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
