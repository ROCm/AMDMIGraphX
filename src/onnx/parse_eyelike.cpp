#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_eyelike : op_parser<parse_eyelike>
{
    std::vector<op_desc> operators() const { return {{"EyeLike"}}; }

    instruction_ref parse(const op_desc&,
                          const onnx_parser&,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto input_shape = args[0]->get_shape();
        auto input_lens  = input_shape.lens();
        if(input_lens.size() != 2)
        {
            // only rank 2 tensors, error handling?
        }

        // ONNX doc says to use types in order of attribute > input tensor type > default float
        // I think shape defaults to float
        shape::type_t output_type = args[0]->get_shape().type();
        if(contains(info.attributes, "dtype"))
        {
            output_type = get_type(info.attributes.at("dtype").i());
        }

        int k = 0;
        if(contains(info.attributes, "k"))
        {
            k = info.attributes.at("k").i();
        }

        auto num_rows = input_lens.front();
        auto num_cols = input_lens.back();
        std::vector<float> eyelike_mat(num_rows * num_cols); // setting type?
        for(int i = 0; i < num_rows; ++i)
        {
            for(int j = 0; j < num_cols; ++j)
            {
                if(i == (j + k))
                {
                    eyelike_mat[2 * i + j] = 1.;
                }
                else
                {
                    eyelike_mat[2 * i + j] = 0.;
                }
            }
        }

        auto eyelike =
            info.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                                 info.add_literal(migraphx::literal{
                                     migraphx::shape{output_type, input_lens}, eyelike_mat}));
        return eyelike;
    }
}
