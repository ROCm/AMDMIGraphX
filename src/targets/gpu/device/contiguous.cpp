
#include <migraph/gpu/device/contiguous.hpp>
#include <migraph/gpu/device/nary.hpp>

namespace migraph {
namespace MIGRAPH_INLINE_NS {
namespace gpu {
namespace device {

void contiguous(hipStream_t stream, argument result, argument arg)
{
    nary_nonstandard(stream, std::move(result), std::move(arg))([](auto x) { return x; });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraph
