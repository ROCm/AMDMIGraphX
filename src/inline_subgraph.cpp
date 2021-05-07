#include <migraphx/inline_subgraph.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void inline_subgraph::apply(module& p) const
{
    for(auto ins : iterator_for(p))
    {
        if(ins->name() != "if")
            continue;

        auto arg_cond          = ins->inputs().front()->eval();
        const auto& mod_inputs = ins->module_inputs();
        // condition is not constant, but both subgraph outputs
        // are constant, so we can replace each subgraph with
        // a literal
        if(arg_cond.empty())
        {
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
                for(const auto& out : mod_outputs)
                {
                    auto mod_out = out->eval();
                    if(mod_out.empty())
                        return;
                    arg_outs.push_back(mod_out);
                }
            }

            auto out_num            = arg_outs.size() / 2;
            auto cond               = ins->inputs().front();
            const auto& ins_outputs = ins->outputs();
            auto icond              = p.insert_instruction(
                ins, make_op("convert", {{"target_type", shape::float_type}}), cond);
            for(std::size_t i = 0; i < out_num; ++i)
            {
                auto l0 = p.add_literal(literal(arg_outs.at(i).get_shape(), arg_outs.at(i).data()));
                auto l1 = p.add_literal(
                    literal(arg_outs.at(i + out_num).get_shape(), arg_outs.at(i + out_num).data()));
                auto lens  = l0->get_shape().lens();
                auto mcond = p.insert_instruction(
                    ins, make_op("multibroadcast", {{"output_lens", lens}}), icond);
                std::size_t elem_num = l0->get_shape().elements();
                std::vector<float> vec_offset(elem_num, elem_num);
                std::vector<float> vec_ind(elem_num);
                shape sidx{shape::float_type, lens};
                std::iota(vec_ind.begin(), vec_ind.end(), elem_num);
                auto lidx    = p.add_literal(literal(sidx, vec_ind));
                auto loffset = p.add_literal(literal(sidx, vec_offset));
                auto moffset = p.insert_instruction(ins, make_op("mul"), mcond, loffset);
                auto f_ind   = p.insert_instruction(ins, make_op("sub"), lidx, moffset);
                auto ins_ind = p.insert_instruction(
                    ins, make_op("convert", {{"target_type", shape::int32_type}}), f_ind);
                auto l01 = p.insert_instruction(ins, make_op("concat", {{"axis", 0}}), l0, l1);
                auto rl  = p.insert_instruction(
                    ins, make_op("reshape", {{"dims", {l0->get_shape().elements() * 2}}}), l01);
                auto r = p.insert_instruction(ins, make_op("gather", {{"axis", 0}}), rl, ins_ind);
                p.replace_instruction(ins_outputs.at(i), r);
            }
        }
        // cond is constant, inline the corresponding subgraph and discard the other one
        else
        {
            inline_submodule(p, ins);
        }
    }
}

void inline_subgraph::inline_submodule(module& p, instruction_ref ins) const
{

    auto arg_cond = ins->inputs().front()->eval();
    assert(not arg_cond.empty());
    const auto& mod_inputs = ins->module_inputs();
    const auto* smod       = (arg_cond.at<bool>()) ? mod_inputs.at(0) : mod_inputs.at(1);

    std::unordered_map<instruction_ref, instruction_ref> map_ins;
    std::vector<instruction_ref> mod_outputs;
    for(auto sins : iterator_for(*smod))
    {
        instruction_ref copy_ins{};
        if(sins->name() == "@literal")
        {
            auto l   = sins->get_literal();
            copy_ins = p.add_literal(l);
        }
        else if(sins->name() == "@param")
        {
            auto&& name = any_cast<builtin::param>(sins->get_operator()).parameter;
            auto s      = sins->get_shape();
            copy_ins    = p.add_parameter(name, s);
        }
        else if(sins->name() == "@outline")
        {
            auto s   = sins->get_shape();
            copy_ins = p.add_outline(s);
        }
        else
        {
            auto mod_args = sins->module_inputs();
            auto inputs   = sins->inputs();
            std::vector<instruction_ref> copy_inputs(inputs.size());
            std::transform(inputs.begin(), inputs.end(), copy_inputs.begin(), [&](auto i) {

                assert(contains(map_ins, i) or p.has_instruction(i));
                return p.has_instruction(i) ? i : map_ins[i];
            });

            if(sins->name() == "@return")
            {
                mod_outputs = copy_inputs;
                break;
            }

            if(mod_args.empty())
            {
                copy_ins = p.insert_instruction(ins, sins->get_operator(), copy_inputs);
            }
            else
            {
                copy_ins = p.insert_instruction(ins, sins->get_operator(), copy_inputs, mod_args);
            }
        }
        map_ins[sins] = copy_ins;
        mod_outputs   = {copy_ins};
    }

    auto ins_outputs = ins->outputs();
    assert(mod_outputs.size() >= ins_outputs.size());
    for(const auto& out : ins_outputs)
    {
        auto val = out->get_operator().to_value();
        assert(val.contains("index"));
        auto index = val.at("index").to<std::size_t>();
        p.replace_instruction(out, mod_outputs.at(index));
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
