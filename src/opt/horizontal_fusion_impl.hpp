#ifndef MIGRAPHX_GUARD_RTGLIB_HORIZONTAL_FUSION_IMPL_HPP
#define MIGRAPHX_GUARD_RTGLIB_HORIZONTAL_FUSION_IMPL_HPP
#include "common_header.hpp"
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

using Encoder = std::function<unsigned long long(instruction_ref)>;
           
struct hash_value
{
    hash_value()
    {
    };
    std::vector<instruction_ref> instrs;
    std::vector<struct hash_value*> outputs;
    void add_instr(instruction_ref ins)
    {
        instrs.push_back(ins);
    }
};

using hash_value_ptr = hash_value*;           

struct horizontal_fusion_impl
{
    horizontal_fusion_impl(program* p)
        : p_program(p)
    {
        instr2_hash.clear();
        instr2_value.clear();
        register_opcode();
        register_encoder();
    }
    void run();
    void process(instruction_ref ins);
    hash_value_ptr hash(instruction_ref ins);
    hash_value& create_value()
    {
        values.push_back(hash_value{});
        return values[values.size() - 1];
    }
    void add_root(hash_value_ptr ptr)
    {
        root_values.push_back(ptr);
    }
    void register_opcode();
    unsigned short get_opcode(instruction_ref ins);
    void register_encoder();

#ifdef MIGRAPHX_DEBUG_OPT
    void dump_program();
#endif
   private:
    program* p_program;
    std::unordered_map<instruction_ref, bool> instr2_hash;
    std::unordered_map<instruction_ref, hash_value_ptr> instr2_value;
    std::unordered_map<std::string, unsigned short> opcode_table;
    std::unordered_map<std::string, Encoder> op_registry;
    std::vector<hash_value> values;
    std::vector<hash_value_ptr> root_values;
};

unsigned long long EncodeConvBiasRelu(instruction_ref in);
           
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif
