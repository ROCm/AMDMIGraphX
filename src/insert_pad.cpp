#include <migraphx/insert_pad.hpp>
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

void insert_pad::apply(module& p) const
{
    for(auto ins : iterator_for(p))
    {
        const std::string& op_name = ins->name();
        if(op_name != "convolution" and op_name != "im2col" and op_name != "pooling")
            continue;
        auto input = ins->inputs().front();
        if(op_name == "convolution" or op_name == "im2col")
            update_op(input, ins, p);
        else if(op_name == "pooling")
            update_pooling(input, ins, p);
    }
}

void insert_pad::update_op(const instruction_ref& input,
                           const instruction_ref& ins,
                           module& p) const
{
    // auto pad_op = any_cast<op::pad>(input->get_operator());
    // if(!pad_op.symmetric())
    //     return;

    auto op = any_cast<op::convolution>(ins->get_operator());

    auto kdims = input->get_shape().lens().size() - 2;
    if(std::equal(op.padding.begin(),
                  op.padding.begin() + kdims,
                  op.padding.begin() + kdims,
                  op.padding.end()))
        return;

    std::vector<int64_t> padding(input->get_shape().lens().size() * 2, 0);
    std::vector<size_t> pads_l(op.padding.begin(), op.padding.begin() + kdims);
    std::vector<size_t> pads_r(op.padding.begin() + kdims, op.padding.end());
    op.padding = std::vector<size_t>(kdims * 2, 0);
    std::copy(pads_l.begin(), pads_l.end(), padding.begin() + 2);
    std::copy(pads_r.begin(), pads_r.end(), padding.begin() + kdims + 2 + 2);

    auto pad_op = p.insert_instruction(ins, op::pad{padding}, input);

    std::vector<instruction_ref> new_inputs{ins->inputs()};
    new_inputs.front() = pad_op;

    p.replace_instruction(ins, op, new_inputs);
}

void insert_pad::update_pooling(const instruction_ref& input,
                                const instruction_ref& ins,
                                module& p) const
{
    auto op = any_cast<op::pooling>(ins->get_operator());
    // if(op.mode == "average")
    // {
    //     return;
    // }
    auto kdims = input->get_shape().lens().size() - 2;
    if(std::equal(op.padding.begin(),
                  op.padding.begin() + kdims,
                  op.padding.begin() + kdims,
                  op.padding.end()))
        return;

    std::vector<int64_t> padding(input->get_shape().lens().size() * 2, 0);
    std::vector<size_t> pads_l(op.padding.begin(), op.padding.begin() + kdims);
    std::vector<size_t> pads_r(op.padding.begin() + kdims, op.padding.end());
    op.padding = std::vector<size_t>(kdims * 2, 0);
    std::copy(pads_l.begin(), pads_l.end(), padding.begin() + 2);
    std::copy(pads_r.begin(), pads_r.end(), padding.begin() + kdims + 2 + 2);

    float pad_val = ((op.mode == "max") ? std::numeric_limits<float>::lowest() : 0.0f);
    auto pad_op = p.insert_instruction(ins, op::pad{padding, pad_val}, input);

    std::vector<instruction_ref> new_inputs{ins->inputs()};
    new_inputs.front() = pad_op;

    p.replace_instruction(ins, op, new_inputs);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
