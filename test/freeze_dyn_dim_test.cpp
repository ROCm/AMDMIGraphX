/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/freeze_dyn_dim.hpp>
#include <migraphx/split_single_dyn_dim.hpp>
#include <migraphx/program.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <cstdlib>
#include <test.hpp>

// Cross-platform env helpers (Windows has _putenv_s instead of setenv).
namespace {
inline int set_env(const char* name, const char* value)
{
#ifdef _WIN32
    return _putenv_s(name, value);
#else
    return ::setenv(name, value, /*overwrite=*/1);
#endif
}
inline int unset_env(const char* name)
{
#ifdef _WIN32
    return _putenv_s(name, "");
#else
    return ::unsetenv(name);
#endif
}
} // namespace

namespace {
// RAII guard that turns MIGRAPHX_DYN_DIM_FREEZE_TO on for the lifetime of
// the guard and restores the prior value on destruction.
struct freeze_env_guard
{
    static constexpr const char* name = "MIGRAPHX_DYN_DIM_FREEZE_TO";
    std::string prev;
    bool had_prev = false;
    explicit freeze_env_guard(const char* value)
    {
        if(auto* p = std::getenv(name))
        {
            prev     = p;
            had_prev = true;
        }
        set_env(name, value);
    }
    ~freeze_env_guard()
    {
        if(had_prev)
            set_env(name, prev.c_str());
        else
            unset_env(name);
    }
};
} // namespace

static void run_freeze(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::freeze_dyn_dim{}, migraphx::dead_code_elimination{}});
}

// Build a tiny dyn-shape program: parameter `data` of shape (dd, 4)
// followed by `relu`. We deliberately use only single-input pointwise
// ops here so the test exercises freeze without depending on the
// multibroadcast 2-input rank-reconciliation logic, which is a separate
// codepath unrelated to the freeze pass itself.
static migraphx::program make_simple_dyn_program(const migraphx::shape::dynamic_dimension& dd)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {dd, migraphx::shape::dynamic_dimension{4, 4}}};
    auto input    = mm->add_parameter("data", s);
    auto relu_ins = mm->add_instruction(migraphx::make_op("relu"), input);
    mm->add_return({relu_ins});
    return p;
}

// With MIGRAPHX_DYN_DIM_FREEZE_TO=50 on a {1..100} dyn dim, the parameter
// should be rewritten to a fully-static {50, 4} shape and the resulting
// program should be static (no dynamic shapes anywhere).
TEST_CASE(freeze_to_static_replaces_dyn_param)
{
    freeze_env_guard guard{"50"};

    migraphx::shape::dynamic_dimension dd{1, 100};
    auto p = make_simple_dyn_program(dd);
    run_freeze(p);

    auto* mm         = p.get_main_module();
    auto param_shape = mm->get_parameter_shape("data");
    EXPECT(not param_shape.dynamic());
    EXPECT(param_shape.lens() == std::vector<std::size_t>{50, 4});
}

// N outside [min, max] must throw -- silently producing a shape the user
// cannot feed is worse than an error.
TEST_CASE(freeze_to_out_of_range_throws)
{
    freeze_env_guard guard{"999"};

    migraphx::shape::dynamic_dimension dd{1, 100};
    auto p = make_simple_dyn_program(dd);
    EXPECT(test::throws([&] { run_freeze(p); }));
}

// With the env var unset, the pass is a no-op and the dynamic-shape
// parameter remains dynamic so the rest of the pipeline (e.g.
// split_single_dyn_dim) can handle it as before.
TEST_CASE(freeze_to_zero_is_noop)
{
    // Explicitly clear in case a prior test left the var set.
    unset_env("MIGRAPHX_DYN_DIM_FREEZE_TO");

    migraphx::shape::dynamic_dimension dd{1, 100};
    auto p = make_simple_dyn_program(dd);
    run_freeze(p);

    auto* mm         = p.get_main_module();
    auto param_shape = mm->get_parameter_shape("data");
    EXPECT(param_shape.dynamic());
}

// freeze_dyn_dim runs before split_single_dyn_dim in the GPU pipeline.
// After freeze, the program is static, so split_single_dyn_dim should be
// a no-op -- no select_module gets emitted, no dim_N submodules created.
TEST_CASE(freeze_then_split_is_noop)
{
    freeze_env_guard guard{"50"};

    migraphx::shape::dynamic_dimension dd{1, 100};
    auto p = make_simple_dyn_program(dd);
    migraphx::run_passes(p,
                         {migraphx::freeze_dyn_dim{},
                          migraphx::dead_code_elimination{},
                          migraphx::split_single_dyn_dim{},
                          migraphx::dead_code_elimination{}});

    auto* mm = p.get_main_module();
    // No select_module should exist anywhere in main.
    auto select_mod = std::find_if(
        mm->begin(), mm->end(), [&](auto&& ins) { return ins.name() == "select_module"; });
    EXPECT(select_mod == mm->end());
    // No dim_N submodules should have been created.  We can't rely on
    // std::distance over the modules vector (its internal storage isn't
    // contiguous), so count via the iteration interface instead.
    std::size_t module_count = 0;
    for(auto* m : p.get_modules())
    {
        (void)m;
        ++module_count;
    }
    EXPECT(module_count == std::size_t{1});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
