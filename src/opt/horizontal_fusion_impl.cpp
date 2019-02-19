#include "horizontal_fusion_impl.hpp"
#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// Register a single operation.
// 1st arg: operation name. 2nd arg: opcode.   3rd arg: encoding function.
void horizontal_fusion_impl::register_op(std::string name, unsigned short opcode, Encoder func)
{
    assert(opcode < 256);
    opcode_table[name] = opcode;
    op_registry[name] = func;
}
           
// Register operations.
void horizontal_fusion_impl::register_all()
{
    register_op("gpu::conv_bias_relu", 2, EncodeConvBiasRelu);
}

// Get operation encoding.
unsigned short horizontal_fusion_impl::get_opcode(instruction_ref ins)
{
    return (opcode_table.find(ins->name()) == opcode_table.end()) ? 0 : opcode_table[ins->name()];
}

// Encode "conv_bias_relu":
//           
// |----- 16 bits -----|----- 16 bits -----|----- 8 bits -----|----- 8 bits-----|----- 16 bits -----|
// |     opcode        | 1st operand value |   filter size    |  kernel size    |     0x0000        |
unsigned long long EncodeConvBiasRelu(instruction_ref ins, unsigned short opcode)
{
    unsigned long long encode = (opcode << 48);
    
    return 0;
}
           
               
hash_value_ptr horizontal_fusion_impl::hash(instruction_ref ins)
{
    if (op_registry.find(ins->name()) == op_registry.end())
        return nullptr;
    Encoder encode = op_registry.at(ins->name());
    unsigned long long val = encode(ins, get_opcode(ins));
    return nullptr;
}
void horizontal_fusion_impl::process(instruction_ref ins)
{
    // Do not hash literals.
    if (ins->name() == "@literal")
        return;
    if (instr2_hash.find(ins) != instr2_hash.end())
    {
        // Hash this instruction.
        if (hash(ins) != nullptr)
        {
            for (auto output : ins->outputs())
            {
                instr2_hash[output] = true;
            } 
        }

    } else {
        std::unordered_map<std::string, int> op2_cnt;
        bool hash_child = false;
        // Do hash if at least two outputs have same operations.
        for (auto output : ins->outputs())
        {
            const std::string& str = output->name();
            if (op2_cnt.find(str) == op2_cnt.end())
                op2_cnt[str] = 1;
            else {
                op2_cnt[str] += 1;
                hash_child = true;
                break;
            }
        }
        if (hash_child)
        {            
            // Create a value for this instruction.
            hash_value& value = create_value(ins);
            add_root(&value);
            // Flag children to be hashed.
            for (auto output : ins->outputs())
            {
                if (op2_cnt[output->name()] > 1)
                    instr2_hash[output] = true;
            }
        }
    }
}
           
void horizontal_fusion_impl::run()
{
    MIGRAPHX_DEBUG(dump("---Before horizontal fusion---"));
    MIGRAPHX_DEBUG(dump_program());
    std::cout << *p_program << std::endl;
    for (auto ins : iterator_for(*p_program))
    {
        process(ins);
    }
}

#ifdef MIGRAPHX_DEBUG_OPT

void horizontal_fusion_impl::dump_program() { std::cout << *p_program << std::endl; }
#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
