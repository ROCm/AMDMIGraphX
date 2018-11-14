#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {

bool is_reshaper(const std::string& name)
{
    // clang-format off
    static const std::unordered_set<std::string> names = {
        "reshape",
        "transpose",
        // "broadcast",
        "contiguous"
    };
    // clang-format on
    return contains(names, name);
}

void simplify_reshapes::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        if(not is_reshaper(ins->name()))
            continue;
        if(ins->outputs().size() != 1)
            continue;
        if(is_reshaper(ins->outputs().front()->name()))
            continue;
        // Gather reshapes
        std::vector<instruction_ref> reshapes{ins};
        while(is_reshaper(reshapes.back()->name()))
        {
            assert(!reshapes.back()->inputs().empty());
            assert(p.has_instruction(reshapes.back()->inputs().front()));
            reshapes.push_back(reshapes.back()->inputs().front());
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
}

} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx
