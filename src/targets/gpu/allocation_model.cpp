#include <migraphx/gpu/allocation_model.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/module.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

std::string gpu_allocation_model::name() const { return "hip::allocate"; }
operation gpu_allocation_model::allocate(const shape& s) const
{
    return make_op(name(), {{"shape", to_value(s)}});
}

operation gpu_allocation_model::preallocate(const shape& s, const std::string& id) const
{
    return make_op("hip::hip_allocate_memory", {{"shape", to_value(s)}, {"id", id}});
}

std::string gpu_allocation_model::copy() const { return "hip::copy"; }

std::function<instruction_ref(instruction_ref, const shape&)> gpu_allocation_model::allocation_inserter(module& m) const
{
    std::unordered_map<instruction_ref, std::string> prog_output_names{};
    auto last = instruction::get_output_alias(std::prev(m.end()));
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
            prog_output_names[ins] = m.name() + ":#output_" + std::to_string(index++);
        }
    }
    bool offload = this->offload_copy;
    return [=, &m](instruction_ref ins, const shape& s) {
        // Instruction's output is an input of the ret instruction
        if(offload)
        {
            auto result = m.insert_instruction(
                ins, make_op("hip::allocate", {{"shape", to_value(s)}}));
            return result;
        }

        auto ins_alias = instruction::get_output_alias(ins);
        if(last->name() == "@return" and prog_output_names.count(ins_alias) > 0)
        {
            return m.add_parameter(prog_output_names.at(ins_alias), s);
        }
        else if(ins == last)
        {
            return m.add_parameter("output", s);
        }

        return m.insert_instruction(
            ins, make_op("hip::allocate", {{"shape", to_value(s)}}));
    };
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
