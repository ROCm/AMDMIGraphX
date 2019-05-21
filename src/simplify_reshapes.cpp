#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/as_shape.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

bool is_reshaper(instruction_ref ins)
{
    // clang-format off
    static const std::unordered_set<std::string> names = {
        "reshape",
        "contiguous",
        "squeeze",
        "unsqueeze"
    };
    // clang-format on
    return contains(names, ins->name());
}

bool is_transpose_output(instruction_ref ins)
{
    if(ins->outputs().size() != 1)
        return false;
    if(ins->outputs().front()->name() == "contiguous")
        return is_transpose_output(ins->outputs().front());
    return ins->outputs().front()->name() == "transpose";
}

instruction_ref find_transpose_input(instruction_ref ins)
{
    if(ins->inputs().size() != 1)
        return ins;
    if(ins->inputs().front()->name() == "contiguous")
        return find_transpose_input(ins->inputs().front());
    if(ins->inputs().front()->name() == "transpose")
        return ins->inputs().front();
    return ins;
}

void simplify_reshapes::apply(program& p) const
{
    auto end = std::prev(p.end());
    for(auto ins : iterator_for(p))
    {
        if(ins == end and ins->name() == "contiguous")
            continue;
        // Skip possible dead instructions
        if(ins->outputs().empty() and ins != end)
            continue;
        if(is_reshaper(ins))
        {
            if(std::any_of(ins->outputs().begin(), ins->outputs().end(), &is_reshaper))
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
        else if(ins->name() == "transpose")
        {
            if(is_transpose_output(ins))
                continue;
            auto x = ins;
            auto t = ins;
            do
            {
                x = t;
                t = find_transpose_input(x);
            } while(x != t and t->name() == "transpose");
            if(t == ins or t->name() != "transpose")
                continue;
            p.replace_instruction(ins, t->inputs().front());
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
