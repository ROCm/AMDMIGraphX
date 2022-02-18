#include <migraphx/auto_contiguous.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void auto_contiguous::apply(module& p) const
{
    std::string key = "std_shape";
    for(auto ins : reverse_iterator_for(p))
    {
        auto&& attr = ins->get_operator().attributes();
        if((attr.contains(key) and attr.at(key).to<bool>()))
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

    // if ops used as output param are alias 0, add a contiguous for the output
    // so return outputs with standard shape
    if(last->name() == "@return")
    {
        auto inputs = last->inputs();
        for(auto ins : inputs)
        {
            if(ins->name() == "contiguous")
                continue;

            auto ins_alias = ins->get_operator().output_alias({});
            if(ins_alias == 0 and ins->get_shape().element_space() !=
                                      ins->inputs().front()->get_shape().element_space())
            {
                auto cont_ins = p.insert_instruction(last, make_op("contiguous"), ins);
                p.replace_instruction(ins, cont_ins);
            }
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
