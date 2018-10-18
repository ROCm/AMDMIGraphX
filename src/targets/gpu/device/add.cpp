#include <migraph/gpu/device/add.hpp>
#include <migraph/gpu/device/nary.hpp>

namespace migraph {
namespace gpu {
namespace device {

void add(const argument& result, const argument& arg1, const argument& arg2)
{
    nary(result, arg1, arg2)([](auto x, auto y) { return x + y; });
}

void add(const argument& result, const argument& arg1, const argument& arg2, const argument& arg3)
{
    nary(result, arg1, arg2, arg3)([](auto x, auto y, auto z) { return x + y + z; });
}

} // namespace device
} // namespace gpu
} // namespace migraph
