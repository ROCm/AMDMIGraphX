#include <migraphx/tf/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pad_calc.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_pooling : op_parser<parse_pooling>
{
    bool transpose() const { return true; }
    std::vector<op_desc> operators() const { return {{"AvgPool"}, {"MaxPool"}}; }

    instruction_ref parse(const op_desc& opd,
                          const tf_parser& parser,
                          tf_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        op::pooling op{starts_with(opd.tf_name, "Max") ? "max" : "average"};

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
        if(contains(info.attributes, "ksize"))
        {
            std::vector<size_t> ksize;
            copy(info.attributes.at("ksize").list().i(), std::back_inserter(ksize));
            parser.reorder_data(ksize);
            if(ksize.size() != 4)
            {
                MIGRAPHX_THROW("ksize should have 4 values");
            }
            op.lengths[0] = ksize[2];
            op.lengths[1] = ksize[3];
        }

        auto l0 = args[0];
        if(contains(info.attributes, "padding"))
        {
            const std::string& pad_mode = info.attributes.at("padding").s();
            if(pad_mode.find("SAME") != std::string::npos)
            {
                auto input_dims = l0->get_shape().lens();
                std::vector<int64_t> pads(input_dims.size());
                calculate_padding(0, pads, input_dims[2], op.stride[0], 1, op.lengths[0]);
                calculate_padding(1, pads, input_dims[3], op.stride[1], 1, op.lengths[1]);

                if(pads[0] != pads[2] || pads[1] != pads[3])
                {
                    std::vector<int64_t> padding = {0, 0, pads[0], pads[1], 0, 0, pads[2], pads[3]};
                    l0                           = info.add_instruction(
                        migraphx::make_op(
                            "pad",
                            {{"pads", padding}, {"value", std::numeric_limits<float>::lowest()}}),
                        l0);
                }
                else
                {
                    op.padding[0] = pads[0];
                    op.padding[1] = pads[1];
                }
            }
        }
        return info.add_instruction(op, l0);
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
