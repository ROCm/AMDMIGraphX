#include <migraphx/tf/op_parser.hpp>
#include <migraphx/tf/tf_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_squeeze : op_parser<parse_squeeze>
{
    std::vector<op_desc> operators() const { return {{"Squeeze"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const tf_parser& /*parser*/,
                          tf_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        auto input_dims = args[0]->get_shape().lens();
        auto axes       = info.attributes.at("squeeze_dims").list().i();
        std::vector<int64_t> op_axes(axes.begin(), axes.end());

        if(op_axes.empty()) // no squeeze_dims provided, remove any dim that equals 1
        {
            for(size_t i = 0; i < input_dims.size(); i++)
            {
                if(input_dims.at(i) == 1)
                {
                    op_axes.push_back(i);
                }
            }
        }
        return info.add_instruction(make_op("squeeze", {{"axes", op_axes}}),
                                    info.make_contiguous(args[0]));
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
