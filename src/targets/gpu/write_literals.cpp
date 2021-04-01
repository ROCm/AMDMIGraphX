#include <migraphx/gpu/write_literals.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/env.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_COPY_LITERALS)

void write_literals::apply(module& p) const
{
    assert(ctx != nullptr);
    std::size_t n = 0;
    for(auto ins : iterator_for(p))
    {
        if(ins->name() == "@literal")
        {
            if(enabled(MIGRAPHX_COPY_LITERALS{}))
            {
                literal l  = ins->get_literal();
                auto pre   = p.add_literal(l);
                auto alloc = p.insert_instruction(std::next(pre), hip_allocate{l.get_shape()});
                p.replace_instruction(ins, hip_copy_to_gpu{}, pre, alloc);
            }
            else
            {
                std::string id = p.name() + ":@literal:" + std::to_string(n);
                p.replace_instruction(ins, hip_copy_literal{ins->get_literal(), id});
                n++;
            }
        }
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
