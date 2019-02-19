#ifndef MIGRAPHX_GUARD_RTGLIB_HORIZONTAL_FUSION_IMPL_HPP
#define MIGRAPHX_GUARD_RTGLIB_HORIZONTAL_FUSION_IMPL_HPP
#include "common_header.hpp"
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// Nodes representing hashed instructions.
struct hash_value
{
    hash_value(unsigned i) : id (i)
    {
    };

    void add_instr(instruction_ref ins)
    {
        instrs.push_back(ins);
    }

    void add_input(struct hash_value* i) { inputs.push_back(i); }
    void add_output(struct hash_value* o) { outputs.push_back(o); }

    unsigned get_id() { return id; }
    const std::vector<struct hash_value*>& get_inputs() const { return inputs; }
    const std::vector<struct hash_value*>& get_outputs() const { return outputs; }
    const std::vector<instruction_ref>& get_instrs() const { return instrs; }
    
private:
    unsigned id;
    std::vector<struct hash_value*> inputs;
    std::vector<struct hash_value*> outputs;
    std::vector<instruction_ref> instrs;
};

using hash_value_ptr = hash_value*;

//  Instruction encoding information, used to hash instructions.
struct encode_info
{
    encode_info(unsigned long long e) : encoding (e)
    {
    };
    void add_input(hash_value_ptr p) { inputs.push_back(p); }
    unsigned long long get_encoding() const { return encoding; }
    const std::vector<hash_value_ptr>& get_inputs() const { return inputs; }
    private:
    unsigned long long encoding;
    std::vector<hash_value_ptr> inputs;
};


using Ins2Val = std::unordered_map<instruction_ref, hash_value_ptr>;
using Encoder = std::function<encode_info(instruction_ref, unsigned short, Ins2Val&)>;

struct horizontal_fusion_impl
{
    horizontal_fusion_impl(program* p)
        : p_program(p)
    {
        instr2_hash.clear();
        instr2_value.clear();
        encode2_value.clear();
        register_all();
    }
    void run();
    void process(instruction_ref ins);
    hash_value_ptr hash(instruction_ref ins);
    void add_root(hash_value_ptr ptr)
    {
        root_values.push_back(ptr);
    }

    unsigned short get_opcode(instruction_ref ins);
    hash_value& get_value(unsigned id) { return values[id]; }

    hash_value& create_value(instruction_ref ins)
    {
        unsigned id = static_cast<unsigned>(values.size());
        values.push_back(hash_value{id});
        hash_value& val = get_value(id);
        val.add_instr(ins);
        instr2_value[ins] = &val;
        return val;
    }

    void register_op(std::string, unsigned short, Encoder);
    void register_all();
    
#ifdef MIGRAPHX_DEBUG_OPT
    void dump_program();
#endif
   private:
    program* p_program;
    // Flag an instruction to hash.
    std::unordered_map<instruction_ref, bool> instr2_hash;
    // Map an instruction to a hash value pointer.
    Ins2Val instr2_value;
    // Map an encoding to a hash value pointer.
    std::unordered_map<unsigned long long, hash_value_ptr> encode2_value;
    // Map an operation name to its encoding.
    std::unordered_map<std::string, unsigned short> opcode_table;
    // Map an operation name to its encoder fucntion.
    std::unordered_map<std::string, Encoder> op_registry;
    // Universe of hash values.
    std::vector<hash_value> values;
    // A collection of root nodes in the hash_value tree.
    std::vector<hash_value_ptr> root_values;
};

// Encoding functions.
encode_info EncodeConvBiasRelu(instruction_ref in, unsigned short opcode, Ins2Val instr2_value);
           
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif
