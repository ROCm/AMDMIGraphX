#include <migraphx/config.hpp>
#include <migraphx/cpu/pointwise.hpp>
#include <migraphx/op/add.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

template struct cpu_binary<op::add>;

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
