
#include <migraph/gpu/device/contiguous.hpp>
#include <migraph/gpu/device/nary.hpp>

namespace migraph {
inline namespace version_1 {
namespace gpu {
namespace device {

void contiguous(hipStream_t stream, argument result, argument arg)
{
    nary_nonstandard(stream, std::move(result), std::move(arg))([](auto x) { return x; });
}

} // namespace device
} // namespace gpu
} // namespace version_1
} // namespace migraph
