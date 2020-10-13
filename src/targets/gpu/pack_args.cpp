#include <migraphx/gpu/pack_args.hpp>
#include <migraphx/requires.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

std::vector<char> pack_args(const std::vector<kernel_argument>& args)
{
    std::vector<char> kernargs;
    for(auto&& arg : args)
    {
        std::size_t n = arg.size;
        const auto* p = static_cast<const char*>(arg.data);
        // Insert padding
        std::size_t padding = (arg.align - (kernargs.size() % arg.align)) % arg.align;
        kernargs.insert(kernargs.end(), padding, 0);
        kernargs.insert(kernargs.end(), p, p + n);
    }
    return kernargs;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
