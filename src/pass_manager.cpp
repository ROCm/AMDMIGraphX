#include "migraphx/module_ref.hpp"
#include <migraphx/program.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/target.hpp>
#include <migraphx/env.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/time.hpp>
#include <migraphx/iterator_for.hpp>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void validate_pass(module& mod, const pass& p, tracer trace)
{
    (void)mod;
    (void)p;
    (void)trace;
#ifndef NDEBUG
    trace("Validate ...");
    auto invalid = mod.validate();
    if(invalid != mod.end())
    {
        auto index = std::distance(mod.begin(), invalid);
        MIGRAPHX_THROW(p.name() + " pass produces invalid program at instruction " +
                       std::to_string(index) + ": " + invalid->name());
    }
    trace();
#endif
}
void run_pass(module& mod, const pass& p, tracer trace)
{
    trace("Module: ", mod.name(), ", Pass: ", p.name());
    assert(mod.validate() == mod.end());
    p.apply(mod);
    trace(mod);
    validate_pass(mod, p, trace);
}
void run_pass(program& prog, const pass& p, tracer trace)
{
    trace("Pass: ", p.name());
    p.apply(prog);
    trace(prog);
}

void run_passes(module& mod, const std::vector<pass>& passes, tracer trace)
{
    for(const auto& p : passes)
    {
        run_pass(mod, p, trace);
    }
}

void run_passes(program& prog, const std::vector<pass>& passes, tracer trace)
{
    for(const auto& p : passes)
    {
        auto mods = prog.get_modules();
        for(const auto& mod : reverse(mods))
        {
            run_pass(*mod, p, trace);
        }
        run_pass(prog, p, trace);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
