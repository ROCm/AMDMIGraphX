#include <migraphx/inline_subgraph.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void inline_subgraph::apply(module& p) const
{
    for(auto ins : iterator_for(p))
    {
        if(ins->name() != "if")
            continue;

        const auto& mod_inputs = ins->module_inputs();
        std::vector<argument> arg_outs;
        for(const auto& mod : mod_inputs)
        {
            auto last = std::prev(mod->end());
            std::vector<instruction_ref> mod_outputs;
            if(last->name() == "@return")
            {
                mod_outputs = last->inputs();
            }
            else
            {
                // no return instruction, last instruction is output
                mod_outputs.push_back(last);
            }

            // only one output is considered for now
            auto out     = mod_outputs.front();
            auto mod_out = out->eval();
            if(mod_out.empty())
            {
                return;
            }
            arg_outs.push_back(mod_out);
        }
        assert(arg_outs.size() == 2);

        auto l0    = p.add_literal(literal(arg_outs.at(0).get_shape(), arg_outs.at(0).data()));
        auto l1    = p.add_literal(literal(arg_outs.at(1).get_shape(), arg_outs.at(1).data()));
        auto lens  = l0->get_shape().lens();
        auto cond  = ins->inputs().front();
        auto icond = p.insert_instruction(ins, make_op("convert", {{"target_type", shape::int32_type}}), cond);
        auto mcond = p.insert_instruction(ins, make_op("multibroadcast", {"output_lens", lens}), icond);

        auto l01   = p.insert_instruction(ins, make_op("concat", {{"axis", 0}}), l0, l1);
        auto rl    = p.insert_instruction(ins, make_op("reshape", {{"dims", {l0->get_shape().elements() * 2}}}), l01);
        auto r     = p.insert_instruction(ins, make_op("gather", {{"axis", 0}}), rl, mcond);
        p.replace_instruction(ins, r);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
