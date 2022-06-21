#include <migraphx/inline_module.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static void inline_submodule(module& m, instruction_ref ins, bool cond)
{
    const auto& mod_inputs = ins->module_inputs();
    module_ref smod        = cond ? mod_inputs.at(0) : mod_inputs.at(1);
    auto mod_outputs       = m.insert_module_instructions(ins, smod);

    auto ins_outputs = ins->outputs();
    assert(mod_outputs.size() >= ins_outputs.size());
    for(const auto& out : ins_outputs)
    {
        auto val = out->get_operator().to_value();
        assert(val.contains("index"));
        auto index = val.at("index").to<std::size_t>();
        m.replace_instruction(out, mod_outputs.at(index));
    }
}

void inline_module::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "if")
            continue;

        auto arg_cond = ins->inputs().front()->eval();
        if(not arg_cond.empty())
        {
            bool cond = arg_cond.at<bool>();
            inline_submodule(m, ins, cond);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
