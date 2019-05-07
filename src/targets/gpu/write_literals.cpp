#include <migraphx/gpu/write_literals.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/env.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_COPY_LITERALS)

struct hip_load_literal
{
    shape s;
    std::size_t n = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.s, "shape"), f(self.n, "id"));
    }

    std::string name() const { return "hip::load_literal"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs}.has(0);
        return s;
    }
    argument compute(context& ctx, const shape&, const std::vector<argument>&) const
    {
        return ctx.literals.at(n);
    }
};

void write_literals::apply(program& p) const
{
    assert(ctx != nullptr);
    for(auto ins : iterator_for(p))
    {
        if(ins->name() == "@literal")
        {
            if(enabled(MIGRAPHX_COPY_LITERALS{}))
            {
                literal l  = ins->get_literal();
                auto pre   = p.add_literal(l);
                auto alloc = p.insert_instruction(std::next(pre), hip_allocate{l.get_shape()});
                p.replace_instruction(ins, hip_copy{}, pre, alloc);
            }
            else
            {
                argument a    = to_gpu(ins->get_literal().get_argument());
                std::size_t n = ctx->literals.size();
                ctx->literals.push_back(a);
                p.replace_instruction(ins, hip_load_literal{a.get_shape(), n});
            }
        }
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
