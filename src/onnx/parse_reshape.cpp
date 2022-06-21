#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_reshape : op_parser<parse_reshape>
{
    std::vector<op_desc> operators() const { return {{"Reshape"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        std::vector<int64_t> dims;
        if(args.size() == 1)
        {
            literal s = parser.parse_value(info.attributes.at("shape"));
            s.visit([&](auto v) { copy(v, std::back_inserter(dims)); });
        }
        if(args.size() == 2)
        {
            auto s = args[1]->eval();
            check_arg_empty(s, "Reshape: dynamic shape is not supported");
            s.visit([&](auto v) { copy(v, std::back_inserter(dims)); });
        }

        return info.add_instruction(make_op("reshape", {{"dims", dims}}),
                                    info.make_contiguous(args[0]));
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
