#include <migraphx/fuse_pointwise.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void create_pointwise_modules(module_pass_manager& mpm)
{
    std::size_t n = 0;
    for(auto ins:iterator_for(mpm.get_module()))
    {
        if (not ins->get_operator().attributes().get("pointwise", false))
            continue;
        auto* pm = mpm.create_module("pointwise" + std::to_string(n++));
        pm->set_bypass();
        std::vector<instruction_ref> inputs;
        std::transform(ins->inputs().begin(), ins->inputs().end(), std::back_inserter(inputs), [&](auto input) {
            return pm->add_parameter("x" + std::to_string(inputs.size()), shape{input->get_shape().type()});
        });
        pm->add_instruction(ins->get_operator(), inputs);

        mpm.get_module().replace_instruction(ins, make_op("pointwise"), ins->inputs(), {pm});
    }
}

void fuse_pointwise::apply(module_pass_manager& mpm) const
{
    create_pointwise_modules(mpm);
    mpm.run_pass(dead_code_elimination{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
