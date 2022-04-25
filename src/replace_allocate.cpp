#include <migraphx/replace_allocate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/op/allocate.hpp>
#include <map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::unordered_map<instruction_ref, std::string> create_output_names(module& mod)
{
    std::unordered_map<instruction_ref, std::string> prog_output_names{};
    auto last = instruction::get_output_alias(std::prev(mod.end()));
    if(last->name() == "@return")
    {
        const auto& prog_outputs = last->inputs();
        std::vector<instruction_ref> outputs_alias(prog_outputs.size());

        std::transform(prog_outputs.begin(),
                       prog_outputs.end(),
                       outputs_alias.begin(),
                       [](const auto& i) { return instruction::get_output_alias(i); });

        std::size_t index = 0;
        for(auto ins : outputs_alias)
        {
            prog_output_names[ins] = mod.name() + ":#output_" + std::to_string(index++);
        }
    }
    return prog_output_names;
}

void insert_if_allocations(instruction_ref ins, module& mod, const std::string& model_name)
{
    std::vector<instruction_ref> inputs = ins->inputs();
    std::vector<module_ref> mod_args    = ins->module_inputs();

    std::map<std::string, shape> name_shapes;
    for(const auto& smod : mod_args)
    {
        auto ps = smod->get_parameter_shapes();
        name_shapes.insert(ps.begin(), ps.end());
    }

    bool ins_output_allocated = false;
    for(auto& pn : name_shapes)
    {
        const auto& s = pn.second;
        instruction_ref output{};
        if(s == ins->get_shape() and not ins_output_allocated)
        {
            output = mod.insert_instruction(ins, make_op(model_name, {{"shape", to_value(s)}}));
            ins_output_allocated = true;
        }
        else
        {
            output = mod.insert_instruction(ins, make_op(model_name, {{"shape", to_value(s)}}));
        }
        inputs.push_back(output);
    }
    mod.replace_instruction(ins, ins->get_operator(), inputs, mod_args);
}

void replace_allocate::apply(module& p) const
{
    auto prog_output_names = create_output_names(p);
    auto last              = instruction::get_output_alias(std::prev(p.end()));
    bool main_offload_copy = p.name() == "main" ? this->offload_copy : false;
    std::string model_name = model.name();
    for(auto ins : iterator_for(p))
    {
        if(ins->get_operator().name() == "if")
        {
            insert_if_allocations(ins, p, model_name);
            continue;
        }
        if(ins->get_operator().name() != "allocate")
            continue;
        auto op = ins->get_operator();
        auto v  = op.to_value();
        assert(v.contains("tag"));
        auto tag = v.at("tag").get_string();
        auto s   = ins->get_shape();

        auto ins_alias = instruction::get_output_alias(ins->outputs().front());
        instruction_ref out_param;
        if(not main_offload_copy and last->name() == "@return" and tag.empty() and
           prog_output_names.count(ins_alias) > 0)
        {
            out_param = p.add_parameter(prog_output_names[ins_alias], s);
            p.replace_instruction(ins, out_param);
            continue;
        }
        else if(not main_offload_copy and ins_alias == last and tag.empty())
        {
            out_param = p.add_parameter("output", s);
            p.replace_instruction(ins, out_param);
            continue;
        }
        instruction_ref alloc_ins;
        if(model_name == "cpu::allocate")
            alloc_ins = p.insert_instruction(ins, make_op(model_name, {{"shape", to_value(s)}}));
        else
            alloc_ins = p.insert_instruction(
                ins, make_op(model_name, {{"shape", to_value(s)}, v.at("tag")}));
        p.replace_instruction(ins, alloc_ins);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
