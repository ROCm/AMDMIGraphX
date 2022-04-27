#include <migraphx/gpu/prefuse_ops.hpp>
#include <migraphx/match/layernorm.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

namespace {
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

        auto a = m.insert_instruction(
            ins, make_op("hip::allocate", {{"shape", to_value(x_ins->get_shape())}}));
        m.replace_instruction(ins, make_op("gpu::layernorm"), x_ins, a);
    }
};

struct find_triaddlayernorm
{
    auto matcher() const
    {
        auto add1 =
            match::name("add")(match::none_of(match::is_constant()),
                               match::args(match::any().bind("z1"), match::any().bind("z2")));
        auto add2 = match::name("add")(match::either_arg(0, 1)(add1, match::any().bind("z3")));
        return match::layernorm()(match::var("x")(add2));
    }

    void apply(module& m, match::matcher_result r) const
    {
        auto ins   = r.result;
        auto x_ins = r.instructions["z1"];
        auto y_ins = r.instructions["z2"];
        auto z_ins = r.instructions["z3"];

        for(auto* pins : {&x_ins, &y_ins, &z_ins})
        {
            if(not(*pins)->get_shape().standard())
                *pins = m.insert_instruction(ins, make_op("contiguous"), *pins);
        }

        auto relements = x_ins->get_shape().lens().back();

        if(relements > 1024 or (relements % 4 != 0 and relements > 256))
            return;

        auto a = m.insert_instruction(
            ins, make_op("hip::allocate", {{"shape", to_value(x_ins->get_shape())}}));
        m.replace_instruction(ins, make_op("gpu::triadd_layernorm"), x_ins, y_ins, z_ins, a);
    }
};
} // namespace

void prefuse_ops::apply(module& m) const
{
    match::find_matches(m, find_triaddlayernorm{}, find_layernorm{});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
