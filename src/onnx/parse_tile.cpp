#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_tile : op_parser<parse_tile>
{
    std::vector<op_desc> operators() const { return {{"Tile"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        migraphx::argument arg_s = args[1]->eval();
        check_arg_empty(arg_s, "PARSE_TILE: dynamic shape is not supported");
        std::vector<std::int64_t> repeats;
        arg_s.visit([&](auto input) { repeats.assign(input.begin(), input.end()); });

        auto l0 = args[0];
        for(int i = 0; i < repeats.size(); i++)
        {
            auto l1 = l0;
            for(int j = 1; j < repeats[i]; j++)
            {
                l0 = info.add_instruction(make_op("concat", {{"axis", i}}), l0, l1);
            }
        }
        return l0;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
