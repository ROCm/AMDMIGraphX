#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>

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

        // if perm is empty, use the default value
        auto n_dim = args.front()->get_shape().lens().size();
        if(perm.empty())
        {
            perm.resize(n_dim);
            std::iota(perm.rbegin(), perm.rend(), 0);
        }

        if(perm.size() != n_dim)
        {
            MIGRAPHX_THROW("PARSE_TRANSPOSE: perm and input have diffferent number of dims!");
        }

        return info.add_instruction(make_op("transpose", {{"permutation", perm}}), args.front());
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
