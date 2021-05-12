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

static void update_op(const instruction_ref& input, const instruction_ref& ins, module& m)
{
    auto op         = ins->get_operator();
    auto val        = op.to_value();
    auto op_padding = val.at("padding").to_vector<size_t>();

    auto kdims = input->get_shape().lens().size() - 2;
    if(std::equal(op_padding.begin(),
                  op_padding.begin() + kdims,
                  op_padding.begin() + kdims,
                  op_padding.end()))
        return;

    std::vector<int64_t> padding(input->get_shape().lens().size() * 2, 0);
    std::vector<size_t> pads_l(op_padding.begin(), op_padding.begin() + kdims);
    std::vector<size_t> pads_r(op_padding.begin() + kdims, op_padding.end());
    op_padding = std::vector<size_t>(kdims * 2, 0);
    op.from_value({{"padding", op_padding}});

    std::copy(pads_l.begin(), pads_l.end(), padding.begin() + 2);
    std::copy(pads_r.begin(), pads_r.end(), padding.begin() + kdims + 2 + 2);

    auto pad_op = m.insert_instruction(ins, op::pad{padding}, input);

    auto new_inputs    = ins->inputs();
    new_inputs.front() = pad_op;

    m.replace_instruction(ins, op, new_inputs);
}

static void update_pooling(const instruction_ref& input, const instruction_ref& ins, module& m)
{
    auto op = any_cast<op::pooling>(ins->get_operator());
    if(op.mode == "average")
    {
        return;
    }
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

    // maxpool uses lowest value for padding
    float pad_val = std::numeric_limits<float>::lowest();
    auto pad_op   = m.insert_instruction(ins, op::pad{padding, pad_val}, input);

    auto new_inputs    = ins->inputs();
    new_inputs.front() = pad_op;

    m.replace_instruction(ins, op, new_inputs);
}

void insert_pad::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        const std::string& op_name = ins->name();
        if(op_name != "convolution" and op_name != "im2col" and op_name != "pooling")
            continue;
        auto input = ins->inputs().front();
        if(op_name == "convolution" or op_name == "im2col")
            update_op(input, ins, m);
        else if(op_name == "pooling")
            update_pooling(input, ins, m);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
