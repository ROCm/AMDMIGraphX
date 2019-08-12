#include <migraphx/rewrite_pooling.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/op/reduce_mean.hpp>
#include <migraphx/program.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void rewrite_pooling::apply(program& prog) const
{
    for(auto ins : iterator_for(prog))
    {
        if (ins->name() != "pooling")
            continue;
        if (ins->get_shape().lens().size() != 4)
            continue;
        if (ins->inputs().empty())
            continue;
        auto&& s = ins->inputs().front()->get_shape();
        auto&& op = any_cast<op::pooling>(ins->get_operator());
        if (op.mode != "average")
            continue;
        if (op.padding[0] != 0 and op.padding[1] != 0)
            continue;
        if (op.stride[0] != 1 and op.stride[1] != 1)
            continue;
        if (s.lens()[2] != op.lengths[0] and s.lens()[3] != op.lengths[1])
            continue;
        prog.replace_instruction(ins, op::reduce_mean{{2, 3}}, ins->inputs().front());
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
