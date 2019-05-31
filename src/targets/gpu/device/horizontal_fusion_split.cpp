#include <hip/hip_runtime.h>
#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/horizontal_fusion_split.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/device/launch.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument split(hipStream_t stream,
               const migraphx::shape& output_shape,
               std::vector<migraphx::argument> args,
               unsigned offset)
{
    auto result           = args.back();
    auto arg0             = args[0];
    auto arg1             = args[1];
    std::size_t nelements = output_shape.elements();
    auto* output          = result.data();
    auto* input           = arg0.data();
    const int* map        = reinterpret_cast<const int*>(arg1.data());
    std::size_t bytes     = output_shape.type_size();
    gs_launch(stream, nelements)([=](auto x) {
        std::size_t map_index = map[x + offset];
        char* output_addr     = output + bytes * x;
        char* input_addr      = input + bytes * map_index;
        std::copy(input_addr, input_addr + bytes, output_addr);
    });
    return args.back();
}
} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
