#include <migraphx/tf/op_parser.hpp>
#include <migraphx/tf/tf_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_transpose : op_parser<parse_transpose>
{
    std::vector<op_desc> operators() const { return {{"Transpose"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const tf_parser& /*parser*/,
                          const tf_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto perm = args[1]->eval().get<int32_t>().to_vector();
        std::vector<int64_t> dims(perm.begin(), perm.end());

        return info.add_instruction(make_op("transpose", {{"dims", dims}}), args.front());
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
