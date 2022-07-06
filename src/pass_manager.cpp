/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
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

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_PASSES);

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
    module* mod           = nullptr;
    tracer* t             = nullptr;
    module* common_parent = nullptr;
    program* prog         = nullptr;

    module_pm(module* pmod = nullptr, tracer* pt = nullptr) : mod(pmod), t(pt) {}

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
    virtual module* get_common_parent() override { return common_parent; }
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
    if(enabled(MIGRAPHX_TRACE_PASSES{}))
        trace = tracer{std::cout};
    for(const auto& p : passes)
    {
        module_pm{&mod, &trace}.run_pass(p);
    }
}

void run_passes(program& prog, const std::vector<pass>& passes, tracer trace)
{
    if(enabled(MIGRAPHX_TRACE_PASSES{}))
        trace = tracer{std::cout};
    std::unordered_set<module_ref> visited;
    for(const auto& p : passes)
    {
        auto mods = prog.get_modules();
        auto tree = prog.get_module_tree();
        visited.clear();
        for(const auto& mod : reverse(mods))
        {
            if(mod->bypass())
                continue;
            if(not visited.insert(mod).second)
                continue;
            module_pm mpm{mod, &trace};
            mpm.prog      = &prog;
            auto parents  = range(tree.equal_range(mod));
            auto nparents = distance(parents);
            if(nparents == 0)
                mpm.common_parent = nullptr;
            else if(nparents == 1)
                mpm.common_parent = parents.begin()->second;
            else
                // Just set common parent to main module when there is muliple parents for now
                // TODO: Compute the common parent
                mpm.common_parent = prog.get_main_module();
            mpm.run_pass(p);
        }
        run_pass(prog, p, trace);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
