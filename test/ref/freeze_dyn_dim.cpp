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

#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <cstdlib>
#include <test.hpp>
#include <cmath>

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

// End-to-end test on the ref target: a dynamic-shape program compiled
// with MIGRAPHX_DYN_DIM_FREEZE_TO=N produces a fully-static program that
// runs and returns the same output as a hand-built static reference at
// the same N.
TEST_CASE(freeze_dyn_dim_ref_end_to_end)
{
    freeze_env_guard guard{"8"};

    // Dyn program: relu(x) where x has shape [dyn(1..16), 4].
    migraphx::program p_dyn;
    {
        auto* mm = p_dyn.get_main_module();
        migraphx::shape s{
            migraphx::shape::float_type,
            {migraphx::shape::dynamic_dimension{1, 16}, migraphx::shape::dynamic_dimension{4, 4}}};
        auto input    = mm->add_parameter("data", s);
        auto relu_ins = mm->add_instruction(migraphx::make_op("relu"), input);
        mm->add_return({relu_ins});
    }
    p_dyn.compile(migraphx::make_target("ref"));

    // Static reference: hand-built relu(x) at the same static shape.
    migraphx::program p_static;
    {
        auto* mm = p_static.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {8, 4}};
        auto input    = mm->add_parameter("data", s);
        auto relu_ins = mm->add_instruction(migraphx::make_op("relu"), input);
        mm->add_return({relu_ins});
    }
    p_static.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-1, 2,   -3,  4,  5,   -6,  7,  -8,  9,   10, -11,
                                  12, 13,  -14, 15, -16, -17, 18, 19,  -20, 21, 22,
                                  23, -24, 25,  26, -27, 28,  29, -30, 31,  -32};
    migraphx::shape input_fixed{migraphx::shape::float_type, {8, 4}};
    migraphx::parameter_map params;
    params["data"] = migraphx::argument(input_fixed, input_data.data());

    auto r_dyn    = p_dyn.eval(params).back();
    auto r_static = p_static.eval(params).back();

    std::vector<float> v_dyn;

    std::vector<float> v_static;
    r_dyn.visit([&](auto out) { v_dyn.assign(out.begin(), out.end()); });
    r_static.visit([&](auto out) { v_static.assign(out.begin(), out.end()); });

    EXPECT(v_dyn.size() == std::size_t{32});
    EXPECT(v_dyn.size() == v_static.size());
    EXPECT(std::equal(v_dyn.begin(), v_dyn.end(), v_static.begin(), [](float a, float b) {
        return std::fabs(a - b) < 1e-6f;
    }));
}

// When the env var is unset, the dyn program compiles to ref and runs at
// any size in [1, 16] -- proving the freeze pass has cleanly no-op'd
// rather than committing the program to a single size.
TEST_CASE(freeze_dyn_dim_ref_unset_is_no_regression)
{
    unset_env("MIGRAPHX_DYN_DIM_FREEZE_TO");

    migraphx::program p_dyn;
    {
        auto* mm = p_dyn.get_main_module();
        migraphx::shape s{
            migraphx::shape::float_type,
            {migraphx::shape::dynamic_dimension{1, 16}, migraphx::shape::dynamic_dimension{4, 4}}};
        auto input    = mm->add_parameter("data", s);
        auto relu_ins = mm->add_instruction(migraphx::make_op("relu"), input);
        mm->add_return({relu_ins});
    }
    p_dyn.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-1, 2, -3, 4, 5, -6, 7, -8};
    migraphx::shape input_fixed{migraphx::shape::float_type, {2, 4}};
    migraphx::parameter_map params;
    params["data"] = migraphx::argument(input_fixed, input_data.data());
    auto r         = p_dyn.eval(params).back();

    std::vector<float> v;
    r.visit([&](auto out) { v.assign(out.begin(), out.end()); });
    EXPECT(v.size() == std::size_t{8});
    for(std::size_t i = 0; i < 8; ++i)
        EXPECT(std::fabs(v[i] - std::max(0.0f, input_data[i])) < 1e-6f);
}
