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
        if(ins->inputs().empty())
            continue;
        auto&& s = ins->inputs().front()->get_shape();
        if(not s.standard())
            continue;
        auto&& op = any_cast<op::pooling>(ins->get_operator());
        if(op.mode != "average")
            continue;
        if (!std::all_of(op.padding.begin(), op.padding.end(), [](auto i) { return i == 0; }))
            continue;
        if (!std::all_of(op.stride.begin(), op.stride.end(), [](auto i) { return i == 1; }))
            continue;
        auto lens = s.lens();
        if (!std::equal(lens.begin() + 2, lens.end(), op.lengths.begin(), op.lengths.end()))
            continue;
        std::int64_t n = s.lens()[0];
        std::int64_t c = s.lens()[1];
        auto reshape =
            prog.insert_instruction(ins, op::reshape{{n * c, -1}}, ins->inputs().front());
        auto pooling = prog.insert_instruction(ins, op::reduce_mean{{1}}, reshape);
        std::vector<int64_t> rsp_lens(lens.size(), 1);
        rsp_lens[0] = n;
        rsp_lens[1] = c;
        prog.replace_instruction(ins, op::reshape{rsp_lens}, pooling);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
