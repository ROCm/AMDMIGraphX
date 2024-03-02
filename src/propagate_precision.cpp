#include <migraphx/propagate_precision.hpp>
#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/eliminate_convert.hpp>
#include <unordered_set>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {
struct precision
{
    shape::type_t type;

    friend bool operator==(const precision& xp, const precision& yp) { return xp.type == yp.type; }
    friend bool operator<(const precision& xp, const precision& yp)
    {
        bool is_less = false;
        shape::visit(xp.type, [&](auto x) {
            shape::visit(yp.type, [&](auto y) {
                if(x.is_integral() != y.is_integral())
                    return;
                if(x.is_integral())
                {
                    if(x.is_unsigned() != y.is_unsigned() and x.size() == y.size())
                        is_less = y.is_unsigned();
                    else
                        is_less = x.size() < y.size();
                }
                else
                {
                    is_less = x.size() < y.size();
                }
            });
        });
        return is_less;
    }
    friend bool operator!=(const precision& xp, const precision& yp) { return not(xp == yp); }
    friend bool operator>(const precision& xp, const precision& yp) { return yp < xp; }
    // This is not totally ordered
    friend bool operator<=(const precision& xp, const precision& yp)
    {
        return (xp < yp) or (xp == yp);
    }
    friend bool operator>=(const precision& xp, const precision& yp)
    {
        return (xp > yp) or (xp == yp);
    }
};
} // namespace

static bool is_pointwise_or_reduce(instruction_ref ins)
{
    return contains(ins->name(), "reduce") or
           ins->get_operator().attributes().get("pointwise", false);
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
        std::unordered_set<instruction_ref> non_scalars;
        std::copy_if(ins->inputs().begin(),
                     ins->inputs().end(),
                     std::inserter(non_scalars, non_scalars.end()),
                     &is_non_scalar_const);
        if(non_scalars.size() == 1)
            return *non_scalars.begin();
    }
    return nullopt;
}

static std::unordered_set<instruction_ref> find_adjacent_inputs(instruction_ref start)
{
    std::unordered_set<instruction_ref> result;
    // Promote inputs
    fix([&](auto self, instruction_ref ins) {
        if(not is_pointwise_or_reduce(ins))
            return;
        if(contains(result, ins))
            return;
        auto next = get_next_input(ins);
        if(not next.has_value())
            return;
        result.insert(ins);
        self(*next);
    })(start->inputs().front());
    return result;
}

static std::unordered_set<instruction_ref> find_adjacent_outputs(instruction_ref start)
{
    std::unordered_set<instruction_ref> result;
    // Promote outputs
    fix([&](auto self, instruction_ref ins) {
        for(auto output : ins->outputs())
        {
            if(not is_pointwise_or_reduce(output))
                continue;
            if(contains(result, output))
                continue;
            auto next = get_next_input(output);
            if(not next.has_value())
                continue;
            if(*next != ins)
                continue;
            result.insert(output);
            self(output);
        }
    })(start);
    return result;
}

template <class Map, class Instructions>
static void
insert_instructions_to_upgrade(Map& m, const Instructions& instructions, shape::type_t t)
{
    for(auto ins : instructions)
    {
        auto it = m.find(ins);
        if(it == m.end())
        {
            m[ins] = t;
        }
        else
        {
            it->second = std::max(precision{t}, precision{it->second}).type;
        }
    }
}

static std::unordered_map<instruction_ref, shape::type_t> find_instruction_to_upgrade(module& m)
{
    std::unordered_map<instruction_ref, shape::type_t> result;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "convert")
            continue;
        auto output = precision{ins->get_shape().type()};
        auto input  = precision{ins->inputs().front()->get_shape().type()};
        if(output.type == shape::type_t::bool_type)
            continue;
        if(input < output)
        {
            insert_instructions_to_upgrade(result, find_adjacent_inputs(ins), output.type);
        }
        else if(input > output)
        {
            insert_instructions_to_upgrade(result, find_adjacent_outputs(ins), input.type);
        }
    }
    return result;
}

void propagate_precision::apply(module_pass_manager& mpm) const
{
    auto upgrade = find_instruction_to_upgrade(mpm.get_module());
    for(const auto& [ins, t] : upgrade)
    {
        auto convert1 = mpm.get_module().insert_instruction(
            std::next(ins), make_op("convert", {{"target_type", ins->get_shape().type()}}), ins);
        mpm.get_module().replace_instruction(ins, convert1);
        std::vector<instruction_ref> inputs;
        std::transform(ins->inputs().begin(),
                       ins->inputs().end(),
                       std::back_inserter(inputs),
                       [&, t = t, ins = ins](auto input) {
                           return mpm.get_module().insert_instruction(
                               ins, make_op("convert", {{"target_type", t}}), input);
                       });
        mpm.get_module().replace_instruction(ins, ins->get_operator(), inputs);
    }
    mpm.run_pass(eliminate_convert{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
