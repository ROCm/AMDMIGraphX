#include "transform.hpp"

#include <migraphx/iterator_for.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

void replace_literals_with_params(program& p)
{
    auto* mm                = p.get_main_module();
    auto existing_names     = mm->get_parameter_names();
    std::size_t literal_idx = 0;
    for(auto ins : iterator_for(*mm))
    {
        if(ins->name() != "@literal")
            continue;
        std::string pname;
        do
        {
            pname = "literal:" + std::to_string(literal_idx++);
        } while(contains(existing_names, pname));
        existing_names.push_back(pname);
        mm->replace_instruction(ins, mm->insert_parameter(ins, pname, ins->get_shape()));
    }
    run_passes(*p.get_main_module(), {migraphx::dead_code_elimination{}});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx
