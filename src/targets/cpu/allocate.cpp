#include <migraphx/config.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/context.hpp>
#include <migraphx/cpu/context.hpp>
#include <migraphx/register_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

struct cpu_allocate : auto_register_op<cpu_allocate>
{
    shape s;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.s, "shape"));
    }

    std::string name() const { return "cpu::allocate"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(0);
        return s;
    }
    argument compute(context&, const shape& output_shape, const std::vector<argument>&) const
    {
        argument result{output_shape};
        return result;
    }
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
