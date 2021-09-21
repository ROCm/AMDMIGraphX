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
void run_pass(program& prog, const pass& p, tracer trace)
{
    trace("Pass: ", p.name());
    p.apply(prog);
    trace(prog);
}

struct module_pm : module_pass_manager
{
    module* mod;
    program* prog;
    tracer* t;

    module_pm(module* pmod = nullptr, program* pprog = nullptr, tracer* pt = nullptr)
        : mod(pmod), prog(pprog), t(pt)
    {
    }

    template <class... Ts>
    void trace(Ts&&... xs) const
    {
        assert(t);
        (*t)(xs...);
    }

    virtual module& get_module() override
    {
        assert(mod);
        return *mod;
    }
    virtual module* create_module(const std::string& name) override
    {
        assert(prog);
        return prog->create_module(name);
    }
    virtual void run_pass(const pass& p) override
    {
        assert(mod);
        trace("Module: ", mod->name(), ", Pass: ", p.name());
        assert(mod->validate() == mod->end());
        p.apply(*this);
        trace(*mod);
        validate_pass(*mod, p, *t);
    }
};

module& get_module(module_pass_manager& mpm) { return mpm.get_module(); }

void run_passes(module& mod, const std::vector<pass>& passes, tracer trace)
{
    for(const auto& p : passes)
    {
        module_pm{&mod, nullptr, &trace}.run_pass(p);
    }
}

void run_passes(program& prog, const std::vector<pass>& passes, tracer trace)
{
    for(const auto& p : passes)
    {
        auto mods = prog.get_modules();
        for(const auto& mod : reverse(mods))
        {
            if(mod->bypass())
                continue;
            module_pm{mod, &prog, &trace}.run_pass(p);
        }
        run_pass(prog, p, trace);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
