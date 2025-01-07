#include "trim.hpp"
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <unordered_map>
#include <unordered_set>

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

static instruction_ref capture_arg(std::unordered_set<instruction_ref>& s, instruction_ref ins)
{
    auto alias = instruction::get_output_alias(ins, true);
    if(alias != ins)
    {
        s.insert(ins);
        return capture_arg(s, alias);
    }
    if(contains({"reshape", "contiguous"}, ins->name()))
    {
        s.insert(ins);
        return capture_arg(s, ins->inputs().front());
    }
    return ins;
}

static instruction_ref add_placeholder(module& m, instruction_ref ins)
{
    if(ins->inputs().empty())
        return ins;
    if(ins->can_eval())
    {
        auto e = ins->eval();
        return m.add_literal(literal{e.get_shape(), e.data()});
    }
    return m.add_parameter("x" + std::to_string(m.get_parameters().size()), ins->get_shape());
}

void trim_module(module& m, std::size_t loc, std::size_t n)
{
    auto last = std::prev(m.end(), loc);
    auto start = std::prev(last, n);
    m.remove_instructions(last, m.end());
    if(n == 0)
        return;
    std::unordered_map<instruction_ref, instruction_ref> map_ins;
    std::unordered_set<instruction_ref> instruction_set;
    auto instructions = range(start, m.end());
    for(instruction_ref ins:iterator_for(instructions))
    {
        instruction_set.insert(ins);
        for(auto input:ins->inputs())
        {
            if(contains(instruction_set, input))
                continue;
            auto arg = capture_arg(instruction_set, input);
            auto placeholder = add_placeholder(m, arg);
            assert(placeholder->get_shape() == arg->get_shape());
            if(placeholder == arg)
                continue;
            instruction_set.insert(placeholder);
            map_ins[arg] = placeholder;
        }
    }
    for(auto[old_ins, new_ins]:map_ins)
        m.replace_instruction(old_ins, new_ins);
    run_passes(m, {dead_code_elimination{}});
    for(auto pins:m.get_parameters())
    {
        if(not pins->outputs().empty())
            continue;
        m.remove_instruction(pins);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx

