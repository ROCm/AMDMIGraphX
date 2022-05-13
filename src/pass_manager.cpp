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

void validate_pass(module& mod, const pass& p, const tracer& trace)
{
    (void)mod;
    (void)p;
    (void)trace;
#ifndef NDEBUG
    trace("Validate...");
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

void run_pass(program& prog, const pass& p, tracer& trace)
{
    auto t_start = std::chrono::high_resolution_clock::now();
    p.apply(prog);
    auto t_end             = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    trace(p.name(),
          "Pass: ",
          p.name(),
          "\n",
          prog,
          "Elapsed Wall Time (ms): ",
          elapsed_time_ms,
          "\n");
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
        auto t_start = std::chrono::high_resolution_clock::now();
        p.apply(*this);
        auto t_end             = std::chrono::high_resolution_clock::now();
        double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        trace(p.name(),
              "Module: ",
              mod->name(),
              ", Pass: ",
              p.name(),
              "\n",
              *mod,
              "Elapsed Wall Time (ms): ",
              elapsed_time_ms);
        validate_pass(*mod, p, *t);
    }
};

module& get_module(module_pass_manager& mpm) { return mpm.get_module(); }

void run_passes(module& mod, const std::vector<pass>& passes, tracer trace)
{
    if(enabled(MIGRAPHX_TRACE_PASSES{}) and not trace.enabled())
        trace = tracer{std::cout};
    for(const auto& p : passes)
    {
        module_pm{&mod, nullptr, &trace}.run_pass(p);
    }
}

void run_passes(program& prog, const std::vector<pass>& passes, tracer trace)
{
    if(enabled(MIGRAPHX_TRACE_PASSES{}) and not trace.enabled())
        trace = tracer{std::cout};

    std::unordered_map<std::string, tracer> module_tracer_map;
    for(const auto& p : passes)
    {
        auto mods = prog.get_modules();
        for(const auto& mod : reverse(mods))
        {
            // Set tracer for module passes, if tracer is set to output to file stream then set name
            // of the dump directory. For file dumps, tracer object internally sets the counter for
            // the individual passes' file dumps.
            if(module_tracer_map.find(mod->name()) == module_tracer_map.end())
            {
                module_tracer_map[mod->name()] =
                    // cppcheck-suppress stlFindInsert
                    trace.fs_enabled() ? tracer{trace.dump_dir + "/" + mod->name()} : trace;
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
