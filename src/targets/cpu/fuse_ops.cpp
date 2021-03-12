#include <migraphx/cpu/fuse_ops.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/value.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/dead_code_elimination.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

MIGRAPHX_PRED_MATCHER(has_post_ops, instruction_ref ins)
{
    auto v = ins->get_operator().to_value();
    return v.contains("post_ops");
}

operation merge_post_ops(const operation& op, const operation& post_op)
{
    auto pv = post_op.to_value();
    auto v  = op.to_value();
    v["post_ops"].push_back({{"algo", pv["algo"]},
                             {"alpha", pv["alpha"].value_or(0.0f)},
                             {"beta", pv["beta"].value_or(0.0f)}});
    auto post_ops = pv.at("post_ops");
    for(const auto& po : post_ops)
        v["post_ops"].push_back(po);
    return make_op(op.name(), v);
}

struct find_post_ops
{
    auto matcher() const
    {
        return match::name("dnnl::eltwise", "dnnl::binary")(match::arg(0)(has_post_ops(), match::used_once()));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins   = r.result;
        auto x_ins = ins->inputs().front();
        auto x     = x_ins->get_operator();

        auto op       = merge_post_ops(x, ins->get_operator());
        auto inputs   = x_ins->inputs();
        inputs.back() = ins->inputs().back();
        if(ins->name() == "dnnl::binary")
            inputs.insert(std::prev(inputs.end()), ins->inputs().at(1));
        auto new_shape = try_compute_shape(op, to_shapes(inputs));
        if(new_shape.empty() or new_shape.front() != ins->get_shape())
            return;
        m.replace_instruction(ins, op, inputs);
    }
};

void fuse_ops::apply(module& m) const
{
    for(std::size_t i = 0; i < 4; i++)
    {
        match::find_matches(m, find_post_ops{});
        dead_code_elimination{}.apply(m);
    }
}

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
