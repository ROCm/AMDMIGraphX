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
            update_pooling(input, ins, p);
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

    auto kdims    = input->get_shape().lens().size() - 2;
    auto kdims_it = pad_op.pads.begin() + 2;

    std::vector<size_t> new_pads(kdims_it, kdims_it + kdims);

    T op       = any_cast<T>(ins->get_operator());
    op.padding = new_pads;

    std::vector<instruction_ref> new_inputs{ins->inputs()};
    new_inputs.front() = input->inputs().front();

    p.replace_instruction(ins, op, new_inputs);
}

void eliminate_pad::update_pooling(const instruction_ref& input,
                                   const instruction_ref& ins,
                                   program& p) const
{
    auto pad_op = any_cast<op::pad>(input->get_operator());
    if(!pad_op.symmetric())
        return;

    auto kdims    = input->get_shape().lens().size() - 2;
    auto kdims_it = pad_op.pads.begin() + 2;

    std::vector<size_t> new_pads(kdims_it, kdims_it + kdims);

    auto op = any_cast<op::pooling>(ins->get_operator());
    if(op.mode == "average")
    {
        return;
    }

    op.padding = new_pads;

    std::vector<instruction_ref> new_inputs{ins->inputs()};
    new_inputs.front() = input->inputs().front();

    p.replace_instruction(ins, op, new_inputs);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
