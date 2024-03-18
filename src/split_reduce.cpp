#include <migraphx/split_reduce.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/functional.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct split_fused_reduce
{
    std::vector<std::int64_t> axes{};
    std::string assign = "assign_none";

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"),
            f(self.assign, "assign"));
    }

    shape compute_shape(const std::vector<shape>& inputs, std::vector<module_ref> mods) const
    {
        if(mods.size() != 1)
            MIGRAPHX_THROW("should have one submodule.");
        const auto* sm = mods.front();
        if(sm->get_output_shapes().size() != 1)
            MIGRAPHX_THROW("Only one output supported");
        auto names = sm->get_parameter_names();
        check_shapes{inputs, *this}.has(names.size()).same_ndims();
        std::sort(names.begin(), names.end());
        auto shapes = sm->get_parameter_shapes();
        // Check dimension matches for each input
        if(not equal(names, inputs, [&](const auto& name, const auto& input) {
               return shapes.at(name).lens() == input.lens();
           }))
            MIGRAPHX_THROW("Dimenstion does not match the submodule.");
        const auto& s = inputs.at(0);
        auto lens     = s.lens();
        if(lens != sm->get_output_shapes().front().lens())
        {
            for(const auto& axis : axes)
            {
                lens[axis] = 1;
            }
        }

        return shape::from_permutation(
            sm->get_output_shapes().front().type(), lens, find_permutation(inputs));
    }

    std::string name() const { return "split_fused_reduce"; }
};
MIGRAPHX_REGISTER_OP(split_fused_reduce);

static bool is_reduce(const instruction& ins) { return contains(ins.name(), "reduce"); }

static std::string param_name(std::size_t i, const std::string& prefix = "x")
{
    return prefix + std::to_string(i);
}

struct module_with_inputs
{
    module mod;
    std::vector<instruction_ref> inputs;
};

static std::pair<module_with_inputs, module_with_inputs>
split_module(module_ref m,
             const std::vector<instruction_ref>& splits,
             const std::vector<instruction_ref>& args)
{
    std::unordered_map<instruction_ref, instruction_ref> param_map;
    auto params = m->get_parameter_names();
    std::sort(params.begin(), params.end());
    std::transform(params.begin(),
                   params.end(),
                   args.begin(),
                   std::inserter(param_map, param_map.begin()),
                   [&](const std::string& name, instruction_ref arg) {
                       return std::make_pair(m->get_parameter(name), arg);
                   });

    std::unordered_set<instruction_ref> selected_instructions;
    fix([&](auto self, const std::vector<instruction_ref>& inputs) {
        for(auto input : inputs)
        {
            if(contains(selected_instructions, input))
                continue;
            selected_instructions.insert(input);
            self(input->inputs());
        }
    })(splits);

    std::vector<instruction_ref> instructions1;
    // TODO: copy_if
    for(auto ins : iterator_for(*m))
    {
        if(not contains(selected_instructions, ins))
            continue;
        instructions1.push_back(ins);
    }

    std::vector<instruction_ref> inputs1;
    for(auto ins : instructions1)
    {
        if(not contains(param_map, ins))
            continue;
        inputs1.push_back(param_map[ins]);
    }
    module m1;
    std::unordered_map<instruction_ref, instruction_ref> map_ins1;
    m1.add_instructions(instructions1, &map_ins1);
    std::vector<instruction_ref> outputs;
    std::transform(splits.begin(),
                   splits.end(),
                   std::back_inserter(outputs),
                   [&](instruction_ref ins) { return map_ins1.at(ins); });
    m1.add_return(outputs);

    std::vector<instruction_ref> instructions2;
    for(auto ins : iterator_for(*m))
    {
        if(contains(selected_instructions, ins))
            continue;
        // Input params can be used in both modules
        std::vector<instruction_ref> input_params;
        // TODO: Use join_inserter
        std::copy_if(ins->inputs().begin(),
                     ins->inputs().end(),
                     std::back_inserter(input_params),
                     [&](instruction_ref input) {
                         if(input->name() != "@param")
                             return false;
                         return not contains(instructions2, input);
                     });
        instructions2.insert(instructions2.end(), input_params.begin(), input_params.end());
        instructions2.push_back(ins);
    }

    std::vector<instruction_ref> inputs2;
    for(auto ins : instructions2)
    {
        if(not contains(param_map, ins))
            continue;
        inputs2.push_back(param_map[ins]);
    }
    module m2;
    std::unordered_map<instruction_ref, instruction_ref> map_ins2;
    std::size_t n = 0;
    for(auto ins : splits)
        map_ins2[ins] = m2.add_parameter(param_name(n++), ins->get_shape().as_standard());
    for(auto ins : iterator_for(*m))
    {
        if(ins->name() != "@param")
            continue;
        if(not contains(instructions2, ins))
            continue;
        map_ins2[ins] = m2.add_parameter(param_name(n++), ins->get_shape().as_standard());
    }
    auto r = m2.add_instructions(instructions2, &map_ins2);
    m2.add_return(r);
    return {{std::move(m1), std::move(inputs1)}, {std::move(m2), std::move(inputs2)}};
}

static std::vector<instruction_ref> find_split(module_ref rm)
{
    std::vector<instruction_ref> result;
    auto reduce_ins = std::find_if(rm->begin(), rm->end(), &is_reduce);
    if(reduce_ins == rm->end())
        return result;
    // Bail if there is more than one reduce for now
    if(std::any_of(std::next(reduce_ins), rm->end(), &is_reduce))
        return result;
    result.push_back(reduce_ins);
    // TODO: Find instructions that are used again in the module
    return result;
}

static std::string assign_op(const std::vector<instruction_ref>& splits)
{
    static std::unordered_map<std::string, std::string> m = {
        {"reduce_sum", "assign_add"},
        {"reduce_mean", "assign_add"},
        {"reduce_prod", "assign_mul"},
        {"reduce_max", "assign_max"},
        {"reduce_min", "assign_min"},
    };
    return m.at(splits.front()->name());
}

void split_reduce::apply(module_pass_manager& mpm) const
{
    for(auto ins : iterator_for(mpm.get_module()))
    {
        if(ins->name() != "fused_reduce")
            continue;
        auto* rm    = ins->module_inputs().front();
        auto splits = find_split(rm);
        if(splits.empty())
            continue;
        auto v    = ins->get_operator().to_value();
        auto axes = v["axes"].to_vector<std::int64_t>();
        // TODO: Check reduction size

        auto mp  = split_module(rm, splits, ins->inputs());
        auto* m1 = mpm.create_module(rm->name() + "_0", std::move(mp.first.mod));
        auto* m2 = mpm.create_module(rm->name() + "_1", std::move(mp.second.mod));
        m1->set_bypass();
        m2->set_bypass();

        // Insert split reduce
        auto split_reduce = mpm.get_module().insert_instruction(
            ins, make_op("split_fused_reduce", {{"axes", axes}, {"assign", assign_op(splits)}}), mp.first.inputs, {m1});

        std::vector<instruction_ref> inputs = {split_reduce};
        inputs.insert(inputs.end(), mp.second.inputs.begin(), mp.second.inputs.end());
        auto param_names = m2->get_parameter_names();
        std::sort(param_names.begin(), param_names.end());

        // TODO: Use get_ins_param_map function
        std::unordered_map<instruction_ref, instruction_ref> param_map;
        std::transform(param_names.begin(),
                       param_names.end(),
                       inputs.begin(),
                       std::inserter(param_map, param_map.begin()),
                       [&](const std::string& name, instruction_ref input) {
                           return std::make_pair(m2->get_parameter(name), input);
                       });
        auto replaced = mpm.get_module().insert_instructions(ins, m2, &param_map);
        assert(replaced.size() == 1);
        mpm.get_module().replace_instruction(ins, replaced.front());
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
