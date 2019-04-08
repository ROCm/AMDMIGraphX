#include <migraphx/quantize_ins.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/stringutils.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

instruction_ref
insert_fp16(program& prog,
            instruction_ref& ins,
            shape::type_t type std::unordered_map<instruction_ref, instruction_ref>& map_fp16)
{
    if(map_fp16.count(ins) > 0)
    {
        return map_fp16[ins];
    }

    assert(ins->get_shape().type() == shape::float_type ||
           ins->get_shape().type() == shape::double_type);
    instruction_ref ins_fp16{};
    if(ins == std::prev(prog.end()))
    {
        ins_fp16 = prog.add_instruction(op::fp_conversion{}, ins);
    }
    else
    {
        ins_fp16 = prog.insert_instruction(std::next(ins), op::fp_conversion{}, ins);
    }
    map_fp16[ins] = ins_fp16;

    return ins_fp16;
}

void quantize_ins(program& prog, const std::vector<std::string>& ins_names)
{
    std::unordered_map<instruction_ref, instruction_ref> map_fp16;
    for(auto ins : iterator_for(prog))
    {
        auto name_it = std::find(ins_name.begin(), ins_name.end(), ins->name());
        if(name_it == ins_name.end())
        {
            continue;
        }

        auto inputs = ins->inputs();
        for(auto input : inputs)
        {
            auto s = input->get_shape();
            if(s.type() == shape::float_type || s.type() == shape::double_type)
            {
                auto input_fp16 = insert_fp16(prog, input, s.type(), map_fp16);
                instruction::replace_argument(ins, input, input_fp16, false);
            }
        }
        ins->recompute_ins_shape();

        if(ins->get_shape().type() == shape::half_type)
        {
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
