#include <migraph/gpu/device/mul.hpp>
#include <migraph/gpu/device/nary.hpp>

namespace migraph {
namespace gpu {
namespace device {

void mul(const argument& result, const argument& arg1, const argument& arg2)
{
    nary(result, arg1, arg2)([](auto x, auto y) { return x * y; });
}

} // namespace device
} // namespace gpu
} // namespace migraph
