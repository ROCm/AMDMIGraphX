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
            auto last = std::prev(mod - end());
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
        auto type  = l0->get_shape().type();
        auto cond  = ins->inputs().front();
        auto icond = p.insert_instruction(make_op("convert", {{"target_type", type}}), cond);
        auto mcond = p.insert_instruction(make_op("multibroadcast", {"output_lens", lens}), icond);

        auto l01   = p.insert_intruction(make_op("sub"), l0, l1);
        auto lcond = p.insert_instruction(make_op("mul"), l01, mcond);
        auto r     = p.insert_instruction(make_op("add"), lcond, l1);
        p.replace_instruction(ins, result);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
