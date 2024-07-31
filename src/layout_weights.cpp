#include <migraphx/layout_weights.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static void transpose_layout(module& m, instruction_ref w)
{
    if(contains({"broadcast", "multibroadcast"}, w->name()))
    {
        transpose_layout(m, w->inputs().front());
    }
    else
    {
        std::vector<std::size_t> perm(w->get_shape().ndim());
        std::iota(perm.begin(), perm.end(), 0);
        std::reverse(perm.end() - 2, perm.end());
        auto layout = m.insert_instruction(std::next(w), make_op("layout", {{"permutation", perm}}), w);
        m.replace_instruction(w, layout);
    }
}

void layout_weights::apply(module& m) const
{
    for(auto ins:iterator_for(m))
    {
        if(not contains({"dot"}, ins->name()))
            continue;
        auto w = ins->inputs()[1];
        if(not w->can_eval())
            continue;
        if(w->outputs().size() != 1)
            continue;
        const auto& strides = w->get_shape().strides();
        if(std::any_of(strides.end() - 2, strides.end(), [](auto s) {
            return s == 0;
        }))
            continue;
        transpose_layout(m, w);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
