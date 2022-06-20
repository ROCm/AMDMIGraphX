#include <migraphx/config.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/context.hpp>
#include <migraphx/cpu/context.hpp>
#include <migraphx/register_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

struct cpu_preallocate : auto_register_op<cpu_preallocate>
{
    shape s;
    std::string id = "";
    argument data;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.s, "shape"), f(self.id, "id"));
    }

    std::string name() const { return "cpu::preallocate"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(0);
        return s;
    }
    argument compute(context&, const shape&, const std::vector<argument>&) const { return data; }
    void finalize(context&, const shape&, const std::vector<shape>&) { data = argument(s); }
    lifetime get_lifetime() const { return lifetime::global; }
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
