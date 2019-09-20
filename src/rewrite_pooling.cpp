#include <migraphx/rewrite_pooling.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/op/reshape.hpp>
#include <migraphx/op/reduce_mean.hpp>
#include <migraphx/program.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void rewrite_pooling::apply(program& prog) const
{
    for(auto ins : iterator_for(prog))
    {
        if(ins->name() != "pooling")
            continue;
        if(ins->get_shape().lens().size() != 4)
            continue;
        if(ins->inputs().empty())
            continue;
        auto&& s = ins->inputs().front()->get_shape();
        if(not s.standard())
            continue;
        auto&& op = any_cast<op::pooling>(ins->get_operator());
        if(op.mode != "average")
            continue;
        if(op.padding[0] != 0 and op.padding[1] != 0)
            continue;
        if(op.stride[0] != 1 and op.stride[1] != 1)
            continue;
        if(s.lens()[2] != op.lengths[0] and s.lens()[3] != op.lengths[1])
            continue;
        std::int64_t n = s.lens()[0];
        std::int64_t c = s.lens()[1];
        auto reshape =
            prog.insert_instruction(ins, op::reshape{{n * c, -1}}, ins->inputs().front());
        auto pooling = prog.insert_instruction(ins, op::reduce_mean{{1}}, reshape);
        prog.replace_instruction(ins, op::reshape{{n, c, 1, 1}}, pooling);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
