#include "horizontal_fusion_impl.hpp"
#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static unsigned opcode_bits = 16;
static unsigned hash_id_bits = 16;
static unsigned filter_bits = 8;
static unsigned kernel_bits = 8;

// Register a single operation.
// 1st arg: operation name. 2nd arg: encoding function.
void horizontal_fusion_impl::register_op(std::string name, Encoder func)
{
    op_registry[name] = func;
}
           
// Register operations.
void horizontal_fusion_impl::register_all()
{
    register_op("gpu::convolution", EncodeConvCommon);
    register_op("gpu::conv_bias_relu", EncodeConvCommon);
    register_op("hip::add_relu", EncodeCommon);
}

static unsigned opcode_shift_count()
{
    return ((sizeof(key_type) * 8) - opcode_bits);
}

static unsigned hash_id_shift_count()
{
    return (opcode_shift_count() - hash_id_bits);
}

static unsigned filter_shift_count()
{
    return (hash_id_shift_count() - filter_bits);
}

static unsigned kernel_shift_count()
{
    return (filter_shift_count() - kernel_bits);
}

// Encode common fields.
//
// |----- 16 bits -----|----- 16 bits -------|----- 32 bits -----|
// |      opcode       | 1st operand hash id |
encode_info EncodeCommon(instruction_ref ins, Ins2Val& instr2_value, unsigned opcode)
{
    if (opcode >= ( 1 << opcode_bits))
        return encode_info(0, false);
    key_type encode = (static_cast<key_type>(opcode) << opcode_shift_count());
    instruction_ref op1 = ins->inputs().front();
    assert(instr2_value.find(op1) != instr2_value.end());
    hash_value_ptr op1_val = instr2_value[op1];
    if (op1_val->id >= ( 1 << hash_id_bits))
        return encode_info(0, false);
    encode |= (static_cast<key_type>(op1_val->id) << hash_id_shift_count());
    encode_info info(encode, true);
    info.add_input(op1_val);
    return info;
}

// Encode common fields in convolution:
//
// |----- 16 bits -----|----- 16 bits -------|----- 8 bits -----|----- 8 bits-----|----- 16 bits -----|
// |     opcode        | 1st operand hash id |   filter size    |  kernel size    |     0x0000        |

encode_info EncodeConvCommon(instruction_ref ins, Ins2Val& instr2_value, unsigned opcode)
{
    encode_info info = EncodeCommon(ins, instr2_value, opcode);
    if (!info.is_valid())
        return info;
    key_type encode = info.get_key();
    instruction_ref op2 = ins->inputs().at(1);
    auto lens = op2->get_shape().lens();
    auto filter = lens[lens.size() - 2];
    auto kernel = lens[lens.size() - 1];
    if ((filter < ( 1 << filter_bits)) && (kernel < ( 1 << kernel_bits)))
    {
        encode |= (filter << filter_shift_count());
        encode |= (kernel << kernel_shift_count());
        info.set_key(encode);
        return info;
    } else {
        return encode_info(0, false);
    }
}

// Hash given instruction.           
hash_value_ptr horizontal_fusion_impl::hash(instruction_ref ins)
{
    if (op_registry.find(ins->name()) == op_registry.end())
        return nullptr;

    Encoder encode_func = op_registry.at(ins->name());
    unsigned opcode = hash_opcode(ins);
    encode_info encode_val = encode_func(ins, instr2_value, opcode);
    if (!encode_val.is_valid())
    {
        std::cout << "warning: value hash fails" << std::endl;
        return nullptr;
    }
    key_type key = encode_val.get_key();
    hash_value_ptr hash_val = nullptr;

    if (encode2_value.find(key) != encode2_value.end()) {
        hash_val = encode2_value[key];
        add_instr(hash_val->id);
        instr2_value[ins] = hash_val;
    } else {
        hash_val = &(create_value(ins));
        encode2_value[key] = hash_val;
    }
    for (auto&& input : encode_val.get_inputs())
    {
        add_input(hash_val->id, input);
        add_output(input->id, hash_val);
    }
    return hash_val;
}

hash_value& horizontal_fusion_impl::create_value(instruction_ref ins)
{
    unsigned id = static_cast<unsigned>(values.size());
    values.push_back(hash_value{id, cur_point});
    hash_value& val = get_value(id);
    add_instr(id);
    instr2_value[ins] = &val;
    return val;
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
            for (auto&& output : ins->outputs())
            {
                instr2_hash[output] = true;
            }
            return;
        }
    }
    std::unordered_map<std::string, int> op2_cnt;
    bool hash_child = false;
    // Do hash if at least two outputs have same operations.
    for (auto&& output : ins->outputs())
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
        for (auto&& output : ins->outputs())
        {
            if (op2_cnt[output->name()] > 1)
                instr2_hash[output] = true;
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
        point2_instr[cur_point] = ins;
        cur_point++;
    }
    dump_hash_tree();
}

#ifdef MIGRAPHX_DEBUG_H_FUSION

void horizontal_fusion_impl::dump_program() { std::cout << *p_program << std::endl; }

void horizontal_fusion_impl::dump_hash_value(hash_value& val)
{
    unsigned id = val.id;
    std::cout << "id: " << id << " @" << val.cur_point;
    if (hash_inputs.find(id) != hash_inputs.end())
    {
        std::cout << " input: ";
        for (auto && input : hash_inputs[id])
            std::cout << " " << input->id;
    }

    if (hash_outputs.find(id) != hash_outputs.end())
    {
        std::cout << " output: ";
        for (auto && output : hash_outputs[id])
            std::cout << " " << output->id;
    }
    if (hash_instrs.find(id) != hash_instrs.end())
    {
        std::cout << " instrs: ";
        for (auto && point : hash_instrs[id])
        {
            std::cout << " @" << point;
        }
    }
    std::cout << std::endl;
}

void horizontal_fusion_impl::dump_hash_tree()
{
    for (auto && val : values)
    {
        dump_hash_value(val);
     }

}
#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
