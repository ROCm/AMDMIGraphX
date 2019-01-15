#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// Reshapers that can't handle nonstandard input shapes
bool is_nonstandard_reshaper(instruction_ref ins)
{
    // clang-format off
    static const std::unordered_set<std::string> names = {
        "reshape"
    };
    // clang-format on
    return contains(names, ins->name()) and ins->inputs().front()->name() == "contiguous";
}

bool is_reshaper(instruction_ref ins)
{
    // clang-format off
    static const std::unordered_set<std::string> names = {
        "reshape",
        "transpose",
        // "broadcast",
        "contiguous"
    };
    // clang-format on
    return contains(names, ins->name()) and not is_nonstandard_reshaper(ins);
}


void simplify_reshapes::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        if(not is_reshaper(ins))
            continue;
        if(ins->outputs().size() != 1)
            continue;
        if(is_reshaper(ins->outputs().front()))
            continue;
        // Gather reshapes
        std::vector<instruction_ref> reshapes{ins};
        while(is_reshaper(reshapes.back()))
        {
            assert(!reshapes.back()->inputs().empty());
            assert(p.has_instruction(reshapes.back()->inputs().front()));
            auto input = reshapes.back()->inputs().front();
            reshapes.push_back(input);
        }

        std::pair<instruction_ref, instruction_ref> r{p.end(), p.end()};
        for(auto start : iterator_for(reshapes))
        {
            auto last = std::find_if(reshapes.rbegin(), reshapes.rend(), [&](auto&& i) {
                return i->get_shape() == (*start)->get_shape() and i != (*start);
            });
            if(last != reshapes.rend())
            {
                r = std::make_pair(*start, *last);
                break;
            }
        }
        if(r.first != r.second)
        {
            p.replace_instruction(r.first, r.second);
        }
    }
    // Replace all reshapes with as_shape
    for(auto ins : iterator_for(p))
    {
        if(ins->name() != "reshape")
            continue;
        p.replace_instruction(ins, op::as_shape{ins->get_shape()}, ins->inputs());
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
