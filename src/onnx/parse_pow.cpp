#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

auto compute_type(shape::type_t tb, shape::type_t te)
{
    static std::unordered_map<shape::type_t, int> op_order = {
        {shape::int8_type, 1},
        {shape::uint8_type, 2},
        {shape::int16_type, 3},
        {shape::uint16_type, 4},
        {shape::int32_type, 5},
        {shape::uint32_type, 6},
        {shape::int64_type, 7},
        {shape::uint64_type, 8},
        {shape::half_type, 9},
        {shape::float_type, 10},
        {shape::double_type, 11}
    };

    if (!contains(op_order, tb) or !contains(op_order, te))
    {
        MIGRAPHX_THROW("PARSE_POW: Input data type not supported!");
    }

    if (op_order[tb] >= op_order[te])
    {
        return tb;
    }
    else
    {
        return te;
    }
}

struct parse_pow : op_parser<parse_pow>
{
    std::vector<op_desc> operators() const { return {{"Pow"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        auto tb   = args[0]->get_shape().type();
        auto te   = args[1]->get_shape().type();

        auto tc = compute_type(tb, te);
        if (tc != tb)
        {
            args[0] = 
                info.add_instruction(make_op("convert", {{"target_type", tc}}), args[0]);
        }

        if (tc != te)
        {
            args[1] = 
                info.add_instruction(make_op("convert", {{"target_type", tc}}), args[1]);
        }

        // return info.add_broadcastable_binary_op("pow", args[0], args[1]);
        auto ret = info.add_broadcastable_binary_op("pow", args[0], args[1]);
        if (tc != tb)
        {
            ret = info.add_instruction(make_op("convert", {{"target_type", tb}}), ret);
        }

        return ret;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
