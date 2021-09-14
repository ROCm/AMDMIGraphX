#include "migraphx/common.hpp"
#include "migraphx/errors.hpp"
#include "migraphx/float_equal.hpp"
#include <cmath>
#include <cstdint>
#include <migraphx/eliminate_data_type.hpp>
#include <migraphx/module.hpp>
#include <migraphx/op/quant_dot.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void eliminate_data_type::apply(module& m) const
{
    static const std::vector<std::string> skip_op_names = {
        "convert", "get_tuple_elem", "if", "loop"};
    for(auto ins : iterator_for(m))
    {
        if(ins->name()[0] == '@')
            continue;
        if(contains(skip_op_names, ins->name()))
            continue;
        auto inputs = ins->inputs();
        std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](auto i) {
            if(types.count(i->get_shape().type()) == 0)
                return i;
            return m.insert_instruction(ins, make_op("convert", {{"target_type", target_type}}), i);
        });
        if(inputs == ins->inputs())
            continue;
        auto op         = ins->get_operator();
        auto attributes = op.attributes();
        auto old_type   = ins->get_shape().type();
        auto val        = op.to_value();
        if(attributes.contains("general_data_type"))
        {
            if(ins->name() == "quant_dot")
            {
                auto alpha = val.at("alpha").to<std::float_t>();
                auto beta  = val.at("beta").to<std::float_t>();
                auto dot_res =
                    migraphx::insert_dot_apply_alpha_beta(m, ins, inputs, alpha, beta);
                auto convert = m.insert_instruction(
                    ins, make_op("convert", {{"target_type", old_type}}), dot_res);
                m.replace_instruction(ins, convert);
                return;
            }
            else
            {
                op = make_op(attributes["general_data_type"].to<std::string>(), val);
            }
        }
        auto out = m.insert_instruction(ins, op, inputs);
        auto convert =
            m.insert_instruction(ins, make_op("convert", {{"target_type", old_type}}), out);
        m.replace_instruction(ins, convert);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
