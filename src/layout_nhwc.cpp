#include <migraphx/layout_nhwc.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/dead_code_elimination.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void transform_convolutions(module& m)
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "convolution")
            continue;
        if(ins->get_shape().lens().size() != 4)
            continue;
        auto args = ins->inputs();
        std::transform(args.begin(), args.end(), args.begin(), [&](auto& i) {
            return m.insert_instruction(ins, make_op("layout", {{"permutation", {0, 2, 3, 1}}}), i);
        });
        auto conv = m.insert_instruction(ins, ins->get_operator(), args);
        auto c    = m.insert_instruction(ins, make_op("contiguous"), conv);
        m.replace_instruction(ins, c);
    }
}

void layout_nhwc::apply(module& m) const
{
    transform_convolutions(m);
    dead_code_elimination{}.apply(m);
    eliminate_contiguous{"contiguous"}.apply(m);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
