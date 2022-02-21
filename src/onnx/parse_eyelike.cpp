#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
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
            MIGRAPHX_THROW("EYELIKE: tensor input not of rank 2");
        }
        auto num_rows = input_lens.front();
        auto num_cols = input_lens.back();

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
        if(k >= 0)
        {
            if(static_cast<size_t>(k) >= num_cols)
            {
                std::ostringstream oss;
                oss << "EYELIKE: positive k out of bounds, k = " << k << " num_cols = " << num_cols;
                MIGRAPHX_THROW(oss.str());
            }
        }
        else
        {
            if(static_cast<size_t>(std::abs(k)) >= num_rows)
            {
                std::ostringstream oss;
                oss << "EYELIKE: negative k out of bounds, k = " << k << " num_rows = " << num_cols;
                MIGRAPHX_THROW(oss.str());
            }
        }

        std::vector<char> eyelike_mat(num_rows * num_cols, 0);
		size_t k_abs = std::abs(k);
		if (k >= 0)
		{
			for(size_t i = 0; i < num_rows; ++i)
			{
				if(i + k_abs < num_cols)
					eyelike_mat[(num_cols + 1) * i + k_abs] = 1.;
			}
		}
		else
		{
			for(size_t i = 0; i < num_rows; ++i)
			{
                if(i >= k_abs) // i - k_abs >= 0
                    eyelike_mat[(num_cols + 1) * i - k_abs] = 1.;
			}
		}
        return info.add_literal(
            migraphx::literal{migraphx::shape{output_type, input_lens}, eyelike_mat});
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
