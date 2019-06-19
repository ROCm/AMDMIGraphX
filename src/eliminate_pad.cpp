#include <migraphx/eliminate_pad.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/im2col.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/op/pad.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void eliminate_pad::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        const std::string& op_name = ins->name();
        if(op_name != "convolution" and op_name != "im2col" and op_name != "pooling")
            continue;
        auto input = ins->inputs().front();
        if(input->name() != "pad")
            continue;
        if(op_name == "convolution")
            update_op(op::convolution{}, input, ins, p);
        else if(op_name == "im2col")
            update_op(op::im2col{}, input, ins, p);
        else if(op_name == "pooling")
            update_op(op::pooling{}, input, ins, p);
    }
}

template <class T>
void eliminate_pad::update_op(T,
                              const instruction_ref& input,
                              const instruction_ref& ins,
                              program& p) const
{
    auto pad_op = any_cast<op::pad>(input->get_operator());
    if(!pad_op.symmetric())
        return;

    std::vector<int64_t> pads = pad_op.pads;
    std::array<size_t, 2> new_pads{static_cast<size_t>(pads[2]), static_cast<size_t>(pads[3])};

    T op = any_cast<T>(ins->get_operator());
    // if(op.padding_mode != op::padding_mode_t::default_)
    //     return;
    op.padding = new_pads;

    std::vector<instruction_ref> new_inputs{ins->inputs()};
    new_inputs.front() = input->inputs().front();

    p.replace_instruction(ins, op, new_inputs);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
