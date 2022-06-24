#include <migraphx/propagate_constant.hpp>
#include <migraphx/program.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/par_for.hpp>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

bool skip_propogate(instruction_ref ins)
{
    if(ins->name() == "contiguous")
        return skip_propogate(ins->inputs().front());
    auto&& s = ins->get_shape();
    if(s.broadcasted() and not s.scalar())
        return true;
    if(s.scalar() and s.elements() != 1)
        return true;
    return false;
}

bool is_const(instruction_ref ins) { return ins->can_eval() and not skip_propogate(ins); }

void propagate_constant::apply(module& m) const
{
    std::unordered_set<instruction_ref> const_instrs;
    auto last = std::prev(m.end());

    // Find instructions that can be evaluated to a literal
    for(auto i : iterator_for(m))
    {
        if(is_const(i) and i != last)
            continue;

        std::copy_if(
            i->inputs().begin(),
            i->inputs().end(),
            std::inserter(const_instrs, const_instrs.begin()),
            [&](const instruction_ref ins) { return is_const(ins) and ins->name() != "@literal"; });
    }

    // Compute literals in parallel
    std::vector<instruction_ref> const_instrs_vec{const_instrs.begin(), const_instrs.end()};
    std::vector<argument> literals(const_instrs_vec.size());
    par_for(const_instrs_vec.size(), 1, [&](const auto i) {
        literals[i] = const_instrs_vec[i]->eval();
    });

    // Replace instructions in m
    for(size_t i = 0; i < const_instrs_vec.size(); i++)
    {
        if(not literals[i].empty())
        {
            assert(literals[i].get_shape() == const_instrs_vec[i]->get_shape());
            auto l = m.add_literal(literals[i].get_shape(), literals[i].data());
            m.replace_instruction(const_instrs_vec[i], l);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
