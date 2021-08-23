#include <migraphx/inline_module.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static void inline_submodule(module& m, instruction_ref ins, bool cond)
{
    const auto& mod_inputs = ins->module_inputs();
    const auto* smod       = cond ? mod_inputs.at(0) : mod_inputs.at(1);

    std::unordered_map<instruction_ref, instruction_ref> map_ins;
    std::vector<instruction_ref> mod_outputs;
    for(auto sins : iterator_for(*smod))
    {
        instruction_ref copy_ins{};
        if(sins->name() == "@literal")
        {
            auto l   = sins->get_literal();
            copy_ins = m.add_literal(l);
        }
        else if(sins->name() == "@param")
        {
            auto&& name = any_cast<builtin::param>(sins->get_operator()).parameter;
            auto s      = sins->get_shape();
            copy_ins    = m.add_parameter(name, s);
        }
        else if(sins->name() == "@outline")
        {
            auto s   = sins->get_shape();
            copy_ins = m.add_outline(s);
        }
        else
        {
            auto mod_args = sins->module_inputs();
            auto inputs   = sins->inputs();
            std::vector<instruction_ref> copy_inputs(inputs.size());
            std::transform(inputs.begin(), inputs.end(), copy_inputs.begin(), [&](auto i) {
                return contains(map_ins, i) ? map_ins[i] : i;
            });

            if(sins->name() == "@return")
            {
                mod_outputs = copy_inputs;
                break;
            }

            copy_ins = m.insert_instruction(ins, sins->get_operator(), copy_inputs, mod_args);
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
        m.replace_instruction(out, mod_outputs.at(index));
    }
}

void inline_module::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "if")
            continue;

        auto arg_cond = ins->inputs().front()->eval();
        if(not arg_cond.empty())
        {
            bool cond = arg_cond.at<bool>();
            inline_submodule(m, ins, cond);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
