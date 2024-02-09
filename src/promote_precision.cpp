#include <migraphx/promote_precision.hpp>
#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/ranges.hpp>
#include <unordered_set>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static bool is_lower_precision(shape::type_t in_type, shape::type_t out_type)
{
    bool is_lower = false;
    shape::visit(in_type, [&](auto in) {
        shape::visit(out_type, [&](auto out) {
            if(in.is_integral() != out.is_integral())
                return;
            if(in.is_integral())
            {
                if(in.is_unsigned() != out.is_unsigned() and in.size() == out.size())
                    is_lower = in.is_unsigned();
                else
                    is_lower = out.size() < in.size();
            }
            else
            {
                is_lower = out.size() < in.size();
            }
        });
    });
    return is_lower;
}

static bool is_pointwise_or_reduce(instruction_ref ins)
{
    return contains(ins->name(), "reduce") && ins->get_operator().attributes().get("pointwise", false);
}
static bool is_non_scalar_const(instruction_ref ins)
{
    return not(ins->get_shape().scalar() and ins->can_eval());
}

static std::optional<instruction_ref> get_next_input(instruction_ref ins)
{
    if(ins->inputs().size() == 1)
        return ins->inputs().front();
    if(ins->inputs().size() > 1)
    {
        auto non_scalar = std::find_if(ins->inputs().begin(), ins->inputs().end(), &is_non_scalar_const);
        if(std::any_of(non_scalar, ins->inputs().end(), &is_non_scalar_const))
            return nullopt;
        if (non_scalar == ins->inputs().end())
            return nullopt;
        return *non_scalar;
    }
    return nullopt;
}

static std::unordered_set<instruction_ref> find_adjacent_operators(instruction_ref start)
{
    std::unordered_set<instruction_ref> result;
    // Promote inputs
    fix([&](auto self, instruction_ref ins) {
        for(auto input:ins->inputs())
        {
            if(not is_pointwise_or_reduce(input))
                continue;
            auto next = get_next_input(input);
            if(not next.has_value())
                continue;
            result.insert(input);
            self(*next);
        }
    })(start);
    // Promote outputs
    fix([&](auto self, instruction_ref ins) {
        for(auto output:ins->outputs())
        {
            if(not is_pointwise_or_reduce(output))
                continue;
            auto next = get_next_input(output);
            if(not next.has_value())
                continue;
            if(*next != output)
                continue;
            result.insert(output);
            self(output);
        }
    })(start);
    return result;
}

static std::unordered_map<instruction_ref, shape::type_t> find_instruction_to_upgrade(module& m, shape::type_t t)
{
    std::unordered_map<instruction_ref, shape::type_t> result;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "convert")
            continue;
        auto output_type = ins->get_shape().type();
        auto input_type = ins->inputs().front()->get_shape().type();
        if(output_type == shape::type_t::bool_type)
            continue;
        if(not is_lower_precision(input_type, output_type))
            continue;
        if(ins->inputs().front()->get_shape().type() != t)
            continue;
    }
    return result;
}

void promote_precision::apply(module_pass_manager& mpm) const {}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
