#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_transpose : op_parser<parse_transpose>
{
    std::vector<op_desc> operators() const { return {{"Transpose"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        std::vector<int64_t> perm{};
        if(contains(info.attributes, "perm"))
        {
            auto&& perm_vals = info.attributes["perm"].ints();
            perm             = std::vector<int64_t>(perm_vals.begin(), perm_vals.end());
        }
        return info.add_instruction(make_op("transpose", {{"dims", perm}}), args.front());
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
