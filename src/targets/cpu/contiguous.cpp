#include <migraphx/config.hpp>
#include <migraphx/cpu/pointwise.hpp>
#include <migraphx/op/contiguous.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

template struct cpu_unary<op::contiguous>;

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
