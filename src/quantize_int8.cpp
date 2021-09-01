#include <migraphx/operation.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/quantize_int8.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/op/capture.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/target.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <numeric>
#include <set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_INT8_QUANTIZATION_PARAMS)

static std::vector<shape::type_t>& get_quantizable_type()
{
    static std::vector<shape::type_t> quantable_types = {
        shape::float_type, shape::double_type, shape::half_type};
    return quantable_types;
}

void quantize_int8_pass::apply(module& m) const // NOLINT
{
    const auto& quantizable_types = get_quantizable_type();
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "capture")
            continue;

        auto op_val = ins->get_operator().to_value();
        assert(op_val.contains("ins_index"));

        auto param_index = op_val.at("ins_index").to<std::size_t>();
        auto param       = quant_params[param_index];

        auto input = ins->inputs().front();
        auto s     = input->get_shape();
        if(contains(quantizable_types, s.type()) and s.type() != shape::int8_type)
        {
            auto zero_point  = m.add_literal(static_cast<int8_t>(param.second));
            auto scale       = m.add_literal(literal({s.type()}, {1.0f / param.first}));
            const auto& lens = s.lens();
            scale =
                m.insert_instruction(ins, make_op("multibroadcast", {{"out_lens", lens}}), scale);
            zero_point = m.insert_instruction(
                ins, make_op("multibroadcast", {{"out_lens", lens}}), zero_point);
            auto q_in =
                m.insert_instruction(ins, make_op("quantizelinear"), input, scale, zero_point);
            auto dq_in =
                m.insert_instruction(ins, make_op("dequantizelinear"), q_in, scale, zero_point);
            m.replace_instruction(ins, dq_in);
        }
    }
}

void capture_arguments_pass::apply(module& m) const // NOLINT
{
    assert(param_index != nullptr);
    for(auto ins : iterator_for(m))
    {
        if(not contains(ins_names, ins->name()))
        {
            continue;
        }

        auto inputs = ins->inputs();
        std::vector<instruction_ref> new_args;
        for(auto input : inputs)
        {
            auto new_in = m.insert_instruction(ins, op::capture{(*param_index)++, f}, input);
            new_args.push_back(new_in);
        }
        m.replace_instruction(ins, ins->get_operator(), new_args);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
