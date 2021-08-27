#include <migraphx/instruction.hpp>
#include <migraphx/module.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/quantize_util.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

const std::vector<shape::type_t>& get_quantizable_type()
{
    static std::vector<shape::type_t> quantable_types = {
        shape::float_type, shape::double_type, shape::half_type, shape::int32_type};
    return quantable_types;
}

instruction_ref insert_quant_ins(module& modl,
                                 const instruction_ref& insert_loc,
                                 instruction_ref& ins,
                                 shape::type_t type,
                                 std::unordered_map<instruction_ref, instruction_ref>& map_ins,
                                 float scale,
                                 float shift)
{
    if(map_ins.count(ins) > 0)
    {
        return map_ins[ins];
    }

    if(ins->name() == "undefined")
    {
        return ins;
    }

    auto ins_s = ins->get_shape();
    assert(contains(get_quantizable_type(), ins_s.type()));
    instruction_ref quant_ins{};
    if(type == shape::int8_type)
    {
        auto zero_point = modl.add_literal(static_cast<int8_t>(shift));
        auto ins_scale  = modl.add_literal(1.0f / scale);
        auto lens       = ins->get_shape().lens();
        ins_scale       = modl.insert_instruction(
            insert_loc, make_op("multibroadcast", {{"out_lens", lens}}), ins_scale);
        zero_point = modl.insert_instruction(
            insert_loc, make_op("multibroadcast", {{"out_lens", lens}}), zero_point);
        quant_ins = modl.insert_instruction(
            insert_loc, make_op("quantizelinear"), ins, ins_scale, zero_point);
    }
    else
    {
        quant_ins =
            modl.insert_instruction(insert_loc, make_op("convert", {{"target_type", type}}), ins);
    }

    map_ins[ins] = quant_ins;

    return quant_ins;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
