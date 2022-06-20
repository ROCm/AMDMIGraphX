#include <migraphx/tf/op_parser.hpp>
#include <migraphx/tf/tf_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_slice : op_parser<parse_slice>
{
    std::vector<op_desc> operators() const { return {{"Slice"}}; }

    // Use a literal instruction to replace the shape since output of
    // shape operator are literals in migraphx
    instruction_ref parse(const op_desc& /*opd*/,
                          const tf_parser& /*parser*/,
                          const tf_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto starts     = args[1]->eval().get<int32_t>().to_vector();
        auto size       = args[2]->eval().get<int32_t>().to_vector();
        auto axes       = args[0]->get_shape().lens();
        size_t num_axes = axes.size();

        std::vector<int64_t> axes_int64(axes.begin(), axes.end());
        std::vector<int64_t> starts_int64(starts.begin(), starts.end());
        std::vector<int64_t> ends(num_axes);
        std::vector<int64_t> op_axes(num_axes);
        std::iota(op_axes.begin(), op_axes.end(), 0);
        for(size_t i = 0; i < num_axes; i++)
        {
            if(size[i] == -1)
                ends[i] = axes_int64[i];
            else
                ends[i] = starts_int64[i] + size[i];
        }
        auto op = make_op("slice", {{"starts", starts_int64}, {"ends", ends}, {"axes", op_axes}});
        return info.add_instruction(op, info.make_contiguous(args[0]));
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
