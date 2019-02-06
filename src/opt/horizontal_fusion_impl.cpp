#include "horizontal_fusion_impl.hpp"
#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

hash_value_ptr horizontal_fusion_impl::hash(instruction_ref ins)
{
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
        for (auto output : ins->outputs())
        {
            const std::string& str = output->name();
            if (op2_cnt.find(str) == op2_cnt.end())
                op2_cnt[str] = 1;
            else {
                op2_cnt[str] += 1;
                hash_child = true;
            }
        }
        if (hash_child)
        {            
            // Create a value for this instruction.
            hash_value& value = create_value();
            value.add_instr(ins);
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
