
#include <migraph/gpu/device/contiguous.hpp>
#include <migraph/gpu/device/unary.hpp>

namespace migraph {
namespace gpu {
namespace device {

void contiguous(argument arg, argument result)
{
    unary_nonstandard(arg, result, [](auto x) { return x; });
}

} // namespace device
} // namespace gpu
} // namespace migraph
