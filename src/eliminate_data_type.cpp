#include "migraphx/common.hpp"
#include "migraphx/errors.hpp"
#include "migraphx/float_equal.hpp"
#include <cmath>
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
        if(attributes.contains("general_data_type"))
        {
            if(ins->name() == "quant_dot")
            {
                // TODO(umang) : rewrite logic to support any alpha or beta values, dot doesn't
                // accept alpha or beta values.
                //              it should make use of dot_apply_alpha_beta to achieve same effect
                float alpha = op.to_value()["alpha"].to<std::float_t>();
                float beta  = op.to_value()["beta"].to<std::float_t>();
                if(!float_equal(alpha, 1) or (!float_equal(beta, 0) and ins->inputs().size() > 2))
                {
                    MIGRAPHX_THROW("Quant_dot can't be converted into dot");
                }
                op = make_op(attributes["general_data_type"].to<std::string>());
            }
            else
            {
                op = make_op(attributes["general_data_type"].to<std::string>(), op.to_value());
            }
        }
        auto old_type = ins->get_shape().type();
        auto out      = m.insert_instruction(ins, op, inputs);
        auto convert =
            m.insert_instruction(ins, make_op("convert", {{"target_type", old_type}}), out);
        m.replace_instruction(ins, convert);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
