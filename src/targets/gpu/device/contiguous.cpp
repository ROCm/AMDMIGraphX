
#include <migraph/gpu/device/contiguous.hpp>
#include <migraph/gpu/device/nary.hpp>

namespace migraph {
namespace gpu {
namespace device {

void contiguous(argument result, argument arg)
{
    nary_nonstandard(std::move(result), std::move(arg))([](auto x) { return x; });
}

} // namespace device
} // namespace gpu
} // namespace migraph
