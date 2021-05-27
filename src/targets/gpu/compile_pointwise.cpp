#include <migraphx/gpu/compile_pointwise.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
const std::string simple_pointwise_increment = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/pointwise.hpp>
#include <args.hpp>

using namespace migraphx;

extern "C" {
__global__ void kernel(${params}) 
{
    rotate_last(${args})([](auto&... private_ps) __device__ {
        make_tensors(private_ps)([](auto... private_xs) __device__ {
            auto private_idx = make_index();
            pointwise(private_idx, ${lambda}, private_xs...);
        });
    })
}
    
}

int main() {}

)__migraphx__";

std::string enum_params(std::size_t count, std::string param)
{
    std::vector<std::string> items(count);
    transform(range(count), items.begin(), [&](auto i) { return param + std::to_string(i); });
    return join_strings(items, ",");
}

operation compile_pointwise(const std::vector<shape>& inputs, const std::string& lambda)
{
    hip_compile_options options;
    options.global = 256 * 1024;
    options.local  = 1024;
    options.inputs = inputs;
    options.output = inputs.back();
    auto src       = interpolate_string(simple_pointwise_increment,
                                  {{"params", enum_params(inputs.size(), "void * private_p")},
                                   {"args", enum_params(inputs.size(), "private_p")},
                                   {"lambda", lambda}});
    return compile_hip_code_object(src, options);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
