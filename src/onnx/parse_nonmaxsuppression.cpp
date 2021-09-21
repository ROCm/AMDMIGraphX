#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

// struct parse_nonmaxsuppression : op_parser<parse_nonmaxsuppression>
// {
//     std::vector<op_desc> operators() const { return {{"NonMaxSuppression"}}; }

//     instruction_ref parse(const op_desc& /*opd*/,
//                           const onnx_parser& parser,
//                           onnx_parser::node_info info,
//                           std::vector<instruction_ref> args) const
//     {
//         int cpb = 0;
//         if(contains(info.attributes, "center_point_box"))
//         {
//             cpb = info.attributes.at("center_point_box").i();
//         }
//         return info.add_instruction(make_op("nonmaxsuppression", {{"center_point_box", cpb}}),
//         args);
//     }
// };

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
