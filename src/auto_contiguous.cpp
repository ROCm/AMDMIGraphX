#include <algorithm>
#include <migraphx/auto_contiguous.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void auto_contiguous::apply(module& p) const
{
    std::string key = "standard_input_shape";
    for(auto ins : reverse_iterator_for(p))
    {
        auto&& attr = ins->get_operator().attributes();
        if((attr.contains(key) and attr.at(key).to<bool>()))
        {
            auto args     = ins->inputs();
            auto new_args = args;
            std::transform(args.begin(), args.end(), new_args.begin(), [&](auto in) {
                return p.replace_instruction(ins, make_op("contiguous"), in);
            });

            if(new_args != args)
            {
                p.replace_instruction(ins, ins->get_operator(), new_args);
            }
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
