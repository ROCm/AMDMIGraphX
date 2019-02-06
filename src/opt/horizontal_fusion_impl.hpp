#ifndef MIGRAPHX_GUARD_RTGLIB_HORIZONTAL_FUSION_IMPL_HPP
#define MIGRAPHX_GUARD_RTGLIB_HORIZONTAL_FUSION_IMPL_HPP
#include "common_header.hpp"
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

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
#ifdef MIGRAPHX_DEBUG_OPT
    void dump_program();
#endif
   private:
    program* p_program;
    std::unordered_map<instruction_ref, bool> instr2_hash;
    std::unordered_map<instruction_ref, hash_value_ptr> instr2_value;
    std::vector<hash_value> values;
    std::vector<hash_value_ptr> root_values;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif
