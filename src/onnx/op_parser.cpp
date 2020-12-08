#include <migraphx/onnx/op_parser.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

std::unordered_map<std::string, onnx_parser::op_func>& op_parser_map()
{
    static std::unordered_map<std::string, onnx_parser::op_func> m; // NOLINT
    return m;
}

void register_op_parser(const std::string& name, onnx_parser::op_func f)
{
    op_parser_map()[name] = std::move(f);
}
onnx_parser::op_func get_op_parser(const std::string& name) { return op_parser_map().at(name); }
std::vector<std::string> get_op_parsers()
{
    std::vector<std::string> result;
    std::transform(op_parser_map().begin(),
                   op_parser_map().end(),
                   std::back_inserter(result),
                   [&](auto&& p) { return p.first; });
    return result;
}

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
