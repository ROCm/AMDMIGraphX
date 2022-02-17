#include <migraphx/auto_contiguous.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void auto_contiguous::apply(module& p) const
{
    std::string key = "require_std_shape";
    for(auto ins : reverse_iterator_for(p))
    {
        auto&& attr = ins->get_operator().attributes();
        if((attr.get(key, false)))
        {
            auto args     = ins->inputs();
            auto new_args = args;
            std::transform(args.begin(), args.end(), new_args.begin(), [&](auto in) {
                if(in->name() == "contiguous")
                {
                    return in;
                }
                return p.insert_instruction(ins, make_op("contiguous"), in);
            });

            if(new_args != args)
            {
                p.replace_instruction(ins, ins->get_operator(), new_args);
            }
        }
    }

    auto last = std::prev(p.end());
    for(auto ins : iterator_for(p))
    {
        // for last instruction that is NOT a return
        if(ins->outputs().empty() and ins != last)
            continue;
        shape s = ins->get_shape();
        if(not s.standard() and s.elements() != 0)
        {
            auto c = p.insert_instruction(std::next(ins), make_op("contiguous"), ins);
            p.replace_instruction(ins, c);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
