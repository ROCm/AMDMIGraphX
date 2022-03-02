#include <migraphx/gpu/prefuse_ops.hpp>
#include <migraphx/match/layernorm.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct find_layernorm
{
    auto matcher() const { return match::layernorm(); }

    void apply(module& m, match::matcher_result r) const
    {
        auto ins   = r.result;
        auto x_ins = r.instructions["x"];

        if(not x_ins->get_shape().standard())
            x_ins = m.insert_instruction(ins, make_op("contiguous"), x_ins);

        auto relements = x_ins->get_shape().lens().back();

        if(relements > 1024 or (relements % 4 != 0 and relements > 256))
            return;

        auto a = m.insert_instruction(ins, make_op("hip::allocate", {"shape", to_value(x_ins->get_shape())}));
        m.replace_instruction(ins, make_op("gpu::layernorm"), x_ins, a);
    }
};

void prefuse_ops::apply(module& m) const
{
    match::find_matches(m, find_layernorm{});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
