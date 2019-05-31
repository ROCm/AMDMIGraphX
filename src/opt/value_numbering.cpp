#include "value_numbering.hpp"

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static unsigned opcode_bits  = 16;
static unsigned hash_id_bits = 16;
static unsigned filter_bits  = 8;
static unsigned kernel_bits  = 8;

const std::unordered_map<std::string, encoder>& value_numbering::get_op_registery()
{
    static std::unordered_map<std::string, encoder> m = {
        {"gpu::convolution", encode_conv_common},
        {"gpu::conv_bias_relu", encode_conv_common},
        {"hip::add_relu", encode_common},
        {"convolution", encode_conv_common},
        {"add", encode_common},
        {"relu", encode_common},
    };
    return m;
}

static unsigned opcode_shift_count() { return ((sizeof(key_type) * 8) - opcode_bits); }

static unsigned hash_id_shift_count() { return (opcode_shift_count() - hash_id_bits); }

static unsigned filter_shift_count() { return (hash_id_shift_count() - filter_bits); }

static unsigned kernel_shift_count() { return (filter_shift_count() - kernel_bits); }

// Encode common fields.
//
// |----- opcode_bits -----|----- hash_id_bits -------|----- xxx bits -----|
// |      opcode           | 1st operand hash id      |
encode_info encode_common(instruction_ref ins, ins2_val& instr2_value, unsigned opcode)
{
    if(opcode >= (static_cast<unsigned>(1) << opcode_bits))
        return encode_info(0, false);
    key_type encode     = (static_cast<key_type>(opcode) << opcode_shift_count());
    instruction_ref op1 = ins->inputs().front();
    if(instr2_value.find(op1) == instr2_value.end())
        return encode_info(0, false);
    hash_value_ptr op1_val = instr2_value[op1];

    if((op1_val == nullptr) || (op1_val->id >= (static_cast<unsigned>(1) << hash_id_bits)))
        return encode_info(0, false);
    encode |= (static_cast<key_type>(op1_val->id) << hash_id_shift_count());
    encode_info info(encode, true);
    info.add_input(op1_val);
    return info;
}

// Encode common fields in convolution:
//
// |----- opcode_bits -----|----- hash_id_bits -------|----- filter_bits -----|-----
// kernel-bits-----|----- xxx bits -----| |     opcode            | 1st operand hash id      |
// filter size         |  kernel size         |      0x0000        |

encode_info encode_conv_common(instruction_ref ins, ins2_val& instr2_value, unsigned opcode)
{
    encode_info info = encode_common(ins, instr2_value, opcode);
    if(!info.is_valid())
        return info;
    key_type encode     = info.get_key();
    instruction_ref op2 = ins->inputs().at(1);
    auto lens           = op2->get_shape().lens();
    auto filter         = lens[lens.size() - 2];
    auto kernel         = lens[lens.size() - 1];
    if((filter < (static_cast<unsigned>(1) << filter_bits)) &&
       (kernel < (static_cast<unsigned>(1) << kernel_bits)))
    {
        encode |= (filter << filter_shift_count());
        encode |= (kernel << kernel_shift_count());
        info.set_key(encode);
        return info;
    }
    else
    {
        return encode_info(0, false);
    }
}

// Hash given instruction.
hash_value_ptr value_numbering::hash(instruction_ref ins)
{
    if(op_registry.find(ins->name()) == op_registry.end())
        return nullptr;

    encoder encode_func    = op_registry.at(ins->name());
    unsigned opcode        = hash_opcode(ins);
    encode_info encode_val = encode_func(ins, instr2_value, opcode);
    if(!encode_val.is_valid())
        return nullptr;

    key_type key            = encode_val.get_key();
    hash_value_ptr hash_val = nullptr;

    if(encode2_value.find(key) != encode2_value.end())
    {
        hash_val = encode2_value[key];
        add_instr(hash_val->id);
        instr2_value[ins] = hash_val;
    }
    else
    {
        hash_val           = &(create_value(ins));
        encode2_value[key] = hash_val;
    }
    for(auto&& input : encode_val.get_inputs())
    {
        add_input(hash_val->id, input);
        add_output(input->id, hash_val);
    }
    return hash_val;
}

hash_value& value_numbering::create_value(instruction_ref ins)
{
    unsigned id = static_cast<unsigned>(values.size());
    values.push_back(hash_value{id, cur_point});
    hash_value& val = get_value(id);
    add_instr(id);
    instr2_value[ins] = &val;
    return val;
}

void value_numbering::process(instruction_ref ins)
{
    // Do not hash literals.
    if(ins->name() == "@literal")
        return;
    if(instr2_hash.find(ins) != instr2_hash.end())
    {
        // Hash this instruction.
        if(hash(ins) != nullptr)
        {
            for(auto&& output : ins->outputs())
            {
                instr2_hash[output] = true;
            }
            return;
        }
    }

    std::unordered_map<std::string, int> op2_cnt;
    bool hash_child = false;
    // Do hash if at least two outputs have same operations.
    for(auto&& output : ins->outputs())
    {
        const std::string& str = output->name();
        if(op2_cnt.find(str) == op2_cnt.end())
            op2_cnt[str] = 1;
        else
        {
            op2_cnt[str] += 1;
            hash_child = true;
            break;
        }
    }
    if(hash_child)
    {
        // Create a value for this instruction.
        create_value(ins);
        // Flag children to be hashed.
        for(auto&& output : ins->outputs())
        {
            if(op2_cnt[output->name()] > 1)
                instr2_hash[output] = true;
        }
    }
}

// update hash tree.
void value_numbering::update_hash_tree(unsigned hash_id)
{
    unsigned first       = *(hash_instrs[hash_id].begin());
    hash_instrs[hash_id] = {first};
}

void value_numbering::run()
{
    MIGRAPHX_DEBUG(dump_program());
    for(auto ins : iterator_for(*p_program))
    {
        ins->id = cur_point;
        process(ins);
        point2_instr[cur_point] = ins;
        cur_point++;
    }
    MIGRAPHX_DEBUG(dump_hash_tree());
}

#ifdef MIGRAPHX_DEBUG_OPT

void value_numbering::dump_program() { std::cout << *p_program << std::endl; }

void value_numbering::dump_hash_value(hash_value& val)
{
    unsigned id = val.id;
    std::cout << "id: " << id << " @" << val.cur_point;
    if(hash_inputs.find(id) != hash_inputs.end())
    {
        std::cout << " input: ";
        for(auto&& input : hash_inputs[id])
            std::cout << " " << input->id;
    }

    if(hash_outputs.find(id) != hash_outputs.end())
    {
        std::cout << " output: ";
        for(auto&& output : hash_outputs[id])
            std::cout << " " << output->id;
    }
    if(hash_instrs.find(id) != hash_instrs.end())
    {
        std::cout << " instrs: ";
        for(auto&& point : hash_instrs[id])
        {
            if(point2_instr.find(point) != point2_instr.end())
            {
                int ins_id = point2_instr[point]->id;
                if(ins_id > 0)
                    std::cout << " (" << ins_id << ")";
            }
            std::cout << " @" << point;
        }
    }
    std::cout << std::endl;
}

void value_numbering::dump_hash_tree()
{
    for(auto&& val : values)
    {
        dump_hash_value(val);
    }
}
#endif
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
