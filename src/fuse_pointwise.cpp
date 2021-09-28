#include <migraphx/fuse_pointwise.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void create_pointwise_modules(module_pass_manager& mpm)
{
    std::size_t n = 0;
    for(auto ins : iterator_for(mpm.get_module()))
    {
        if(not ins->get_operator().attributes().get("pointwise", false))
            continue;
        auto* pm = mpm.create_module("pointwise" + std::to_string(n++));
        pm->set_bypass();
        std::vector<instruction_ref> inputs;
        std::transform(ins->inputs().begin(),
                       ins->inputs().end(),
                       std::back_inserter(inputs),
                       [&](auto input) {
                           return pm->add_parameter("x" + std::to_string(inputs.size()),
                                                    shape{input->get_shape().type()});
                       });
        auto r = pm->add_instruction(ins->get_operator(), inputs);
        pm->add_return({r});

        mpm.get_module().replace_instruction(ins, make_op("pointwise"), ins->inputs(), {pm});
    }
}

instruction_ref get_return(module_ref m)
{
    auto last = std::prev(m->end());
    if (last->name() == "@return")
        return last->inputs().front();
    return last;
}

std::vector<instruction_ref> append_pointwise_module(instruction_ref ins, instruction_ref output)
{
    module_ref pm = ins->module_inputs().at(0);
    module_ref xm = output->module_inputs().at(0);

    auto last = std::prev(pm->end());
    assert(last->name() == "@return");
    assert(last->inputs().size());

    std::vector<instruction_ref> inputs = ins->inputs();
    std::unordered_map<instruction_ref, instruction_ref> map_ins;
    for(auto i:range(output->inputs().size()))
    {
        auto input = output->inputs()[i];
        auto param = xm->get_parameter("x" + std::to_string(i));
        if (input == ins)
        {
            map_ins[param] = last->inputs().front();
        }
        else
        {
            map_ins[param] = pm->add_parameter("x" + std::to_string(inputs.size()), input->get_shape());
            inputs.push_back(input);
        }
    }
    pm->insert_module_instructions(last, xm, map_ins);
    return inputs;
}

void find_pointwise_modules(module& m)
{
    for(auto ins : iterator_for(m))
    {
        if (ins->name() != "pointwise")
            continue;
        if (ins->outputs().empty())
            continue;
        auto it = std::find_if(ins->inputs().begin(), ins->inputs().end(), [&](auto i) {
            return i->name() == "pointwise" and i->outputs().size() == 1;
        });
        if (it == ins->inputs().end())
            continue;
        auto new_inputs = append_pointwise_module(*it, ins);
        m.replace_instruction(*it, (*it)->get_operator(), new_inputs, (*it)->module_inputs());
        m.replace_instruction(ins, *it);
    }
}

void fuse_pointwise::apply(module_pass_manager& mpm) const
{
    create_pointwise_modules(mpm);
    mpm.run_pass(dead_code_elimination{});
    for(int i=0;i<8;i++)
    {
        find_pointwise_modules(mpm.get_module());
        mpm.run_pass(dead_code_elimination{});
    }

}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
