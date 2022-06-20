#include <migraphx/tf/op_parser.hpp>
#include <migraphx/tf/tf_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_expanddims : op_parser<parse_expanddims>
{
    std::vector<op_desc> operators() const { return {{"ExpandDims"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const tf_parser& /*parser*/,
                          const tf_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        std::vector<size_t> input_dims = args[0]->get_shape().lens();
        std::vector<int64_t> new_dims(input_dims.begin(), input_dims.end());
        size_t num_dims = input_dims.size();
        int32_t dim     = args[1]->eval().at<int32_t>();

        if(dim < 0)
        {
            new_dims.insert(new_dims.begin() + (num_dims + dim + 1), 1);
        }
        else
        {
            new_dims.insert(new_dims.begin() + dim, 1);
        }
        return info.add_instruction(make_op("reshape", {{"dims", new_dims}}), args[0]);
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
