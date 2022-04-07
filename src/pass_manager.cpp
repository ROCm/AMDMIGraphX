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
#include <unordered_map>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_PASSES);

void validate_pass(module& mod, const pass& p)
{
    (void)mod;
    (void)p;
#ifndef NDEBUG
    std::cout << "Validate..." << std::endl;
    auto invalid = mod.validate();
    if(invalid != mod.end())
    {
        auto index = std::distance(mod.begin(), invalid);
        MIGRAPHX_THROW(p.name() + " pass produces invalid program at instruction " +
                       std::to_string(index) + ": " + invalid->name());
    }
#endif
}

void run_pass(program& prog, const pass& p, tracer& trace)
{
    p.apply(prog);
    trace(p.name(), prog);
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
        assert(mod->validate() == mod->end());
        p.apply(*this);
        trace(p.name(), *mod);
        validate_pass(*mod, p);
    }
};

module& get_module(module_pass_manager& mpm) { return mpm.get_module(); }

void run_passes(module& mod, const std::vector<pass>& passes, tracer trace)
{
    if(enabled(MIGRAPHX_TRACE_PASSES{}))
        trace = tracer{mod.name() + "_passes"};
    for(const auto& p : passes)
    {
        module_pm{&mod, nullptr, &trace}.run_pass(p);
    }
}

void run_passes(program& prog, const std::vector<pass>& passes, tracer trace)
{
    if(enabled(MIGRAPHX_TRACE_PASSES{}) and not trace.enabled())
        trace = tracer{"passes"};
    auto module_trace = trace;
    std::unordered_map<std::string, tracer> module_tracer_map;
    for(const auto& p : passes)
    {
        auto mods = prog.get_modules();
        for(const auto& mod : reverse(mods))
        {
            if(module_tracer_map.find(mod->name()) != module_tracer_map.end())
            {
                module_tracer_map[mod->name()] = module_trace;
                module_tracer_map[mod->name()].dump_dir += "/" + mod->name();
            }
            if(mod->bypass())
                continue;
            module_pm{mod, &prog, &module_tracer_map[mod->name()]}.run_pass(p);
        }
        run_pass(prog, p, trace);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
