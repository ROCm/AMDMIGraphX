#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

auto compute_type(shape::type_t t1, shape::type_t t2)
{
    const static std::unordered_map<int, int> op_order = {{shape::int8_type, 1},
                                                          {shape::uint8_type, 2},
                                                          {shape::int16_type, 3},
                                                          {shape::uint16_type, 4},
                                                          {shape::int32_type, 5},
                                                          {shape::uint32_type, 6},
                                                          {shape::int64_type, 7},
                                                          {shape::uint64_type, 8},
                                                          {shape::half_type, 9},
                                                          {shape::float_type, 10},
                                                          {shape::double_type, 11}};

    int it1 = t1;
    int it2 = t2;
    if(!contains(op_order, it1) or !contains(op_order, it2))
    {
        MIGRAPHX_THROW("PARSE_POW: Input data type not supported!");
    }

    return ((op_order.at(it1) >= op_order.at(it2)) ? t1 : t2);
}

struct parse_pow : op_parser<parse_pow>
{
    std::vector<op_desc> operators() const { return {{"Pow"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto type_base     = args[0]->get_shape().type();
        auto type_exponent = args[1]->get_shape().type();

        auto type_compute = compute_type(type_base, type_exponent);
        if(type_compute != type_base)
        {
            args[0] =
                info.add_instruction(make_op("convert", {{"target_type", type_compute}}), args[0]);
        }

        if(type_compute != type_exponent)
        {
            args[1] =
                info.add_instruction(make_op("convert", {{"target_type", type_compute}}), args[1]);
        }

        auto ret = info.add_broadcastable_binary_op("pow", args[0], args[1]);
        if(type_compute != type_base)
        {
            ret = info.add_instruction(make_op("convert", {{"target_type", type_base}}), ret);
        }

        return ret;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
