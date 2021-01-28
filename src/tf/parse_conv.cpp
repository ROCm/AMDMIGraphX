#include <migraphx/tf/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pad_calc.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_conv : op_parser<parse_conv>
{
    bool transpose() const { return true; }
    std::vector<op_desc> operators() const { return {{"Conv2D"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const tf_parser& parser,
                          tf_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        op::convolution op;
        if(contains(info.attributes, "strides"))
        {
            std::vector<size_t> stride;
            copy(info.attributes.at("strides").list().i(), std::back_inserter(stride));
            parser.reorder_data(stride);
            if(stride.size() != 4)
            {
                MIGRAPHX_THROW("strides should have 4 values");
            }
            op.stride[0] = stride[2];
            op.stride[1] = stride[3];
        }
        if(contains(info.attributes, "dilations"))
        {
            std::vector<size_t> dilation;
            copy(info.attributes.at("dilations").list().i(), std::back_inserter(dilation));
            parser.reorder_data(dilation);
            if(dilation.size() != 4)
            {
                MIGRAPHX_THROW("dilation should have 4 values");
            }
            op.dilation[0] = dilation[2];
            op.dilation[1] = dilation[3];
        }

        auto weights = parser.to_kcxy(args[1]);
        auto l0      = args[0];
        if(contains(info.attributes, "padding"))
        {
            const std::string& pad_mode = info.attributes.at("padding").s();
            if(pad_mode.find("SAME") != std::string::npos)
            {
                op.padding_mode                 = op::padding_mode_t::same;
                std::vector<size_t> weight_dims = weights->get_shape().lens();
                size_t weight_h                 = weight_dims[2];
                size_t weight_w                 = weight_dims[3];

                auto input_dims = l0->get_shape().lens();
                std::vector<int64_t> pads(input_dims.size());
                calculate_padding(0, pads, input_dims[2], op.stride[0], op.dilation[0], weight_h);
                calculate_padding(1, pads, input_dims[3], op.stride[1], op.dilation[1], weight_w);

                if(pads[0] != pads[2] || pads[1] != pads[3])
                {
                    std::vector<int64_t> padding = {0, 0, pads[0], pads[1], 0, 0, pads[2], pads[3]};
                    l0 = info.add_instruction(migraphx::make_op("pad", {{"pads", padding}}), l0);
                }
                else
                {
                    op.padding[0] = pads[0];
                    op.padding[1] = pads[1];
                }
            }
            else if(pad_mode.find("VALID") != std::string::npos)
            {
                op.padding_mode = op::padding_mode_t::valid;
            }
            else if(pad_mode.find("EXPLICIT") != std::string::npos)
            {
                std::vector<size_t> padding;
                copy(info.attributes.at("explicit_paddings").list().i(),
                     std::back_inserter(padding));
                if(padding.size() != 4)
                {
                    MIGRAPHX_THROW("padding should have 4 values");
                }
                if(padding[0] != padding[2] || padding[1] != padding[3])
                {
                    MIGRAPHX_THROW("migraphx does not support asymetric padding");
                }
                op.padding[0] = padding[0];
                op.padding[1] = padding[1];
            }
        }
        return info.add_instruction(op, {l0, weights});
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
