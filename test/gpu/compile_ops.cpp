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
#include <migraphx/gpu/compile_ops.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/serialize.hpp>
#include <test.hpp>
#include <chrono>
#include <map>
#include <mutex>

struct test_tuned
{
    std::string name() const { return "test::tuned"; }

    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs) const
    {
        return inputs.front();
    }
};
MIGRAPHX_REGISTER_OP(test_tuned);

// The budget each solution was compiled with. The registry calls its own copy of the compiler, and
// calls it from the parallel compile threads.
struct compiled_budgets
{
    std::mutex mutex;
    std::map<std::size_t, migraphx::optional<std::chrono::milliseconds>> budgets;

    static compiled_budgets& get()
    {
        static compiled_budgets log;
        return log;
    }
};

// A tuning compiler for test::tuned with three solutions, none of which compiles
struct test_tuned_compiler : migraphx::gpu::compiler<test_tuned_compiler>
{
    std::vector<std::string> names() const { return {"test::tuned"}; }

    migraphx::optional<migraphx::gpu::tuning_config> get_tuning_config(migraphx::gpu::context&,
                                                                       migraphx::instruction_ref,
                                                                       const migraphx::operation&,
                                                                       bool) const
    {
        return migraphx::gpu::tuning_config{"test::tuned", {0, 1, 2}, ""};
    }

    migraphx::gpu::compiler_replace
    compile(migraphx::gpu::context&,
            migraphx::instruction_ref,
            const migraphx::operation&,
            const migraphx::value& solution,
            migraphx::optional<std::chrono::milliseconds> cpu_budget) const
    {
        auto& log = compiled_budgets::get();
        std::lock_guard<std::mutex> lock(log.mutex);
        log.budgets[solution.to<std::size_t>()] = cpu_budget;
        MIGRAPHX_THROW("Not compilable");
    }
};

TEST_CASE(tuning_compile_budget_spares_first_solution)
{
    compiled_budgets::get().budgets.clear();
    migraphx::gpu::context ctx;
    migraphx::module m;
    auto x      = m.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
    auto output = m.add_parameter("output", {migraphx::shape::float_type, {2, 3}});
    auto tuned  = m.add_instruction(
        migraphx::make_op("gpu::precompile_op",
                          {{"op", migraphx::to_value(migraphx::make_op("test::tuned"))}}),
        x,
        output);
    m.add_return({tuned});

    EXPECT(test::throws<migraphx::exception>(
        [&] {
            migraphx::run_passes(
                m,
                {migraphx::gpu::compile_ops{
                    .ctx = &ctx, .tuning_compile_budget = std::chrono::milliseconds{1}}});
        },
        "No valid tuned compilation"));
    std::map<std::size_t, migraphx::optional<std::chrono::milliseconds>> expected = {
        {0, migraphx::nullopt},
        {1, std::chrono::milliseconds{1}},
        {2, std::chrono::milliseconds{1}}};
    EXPECT(compiled_budgets::get().budgets == expected);
}

TEST_CASE(tuning_compile_budget_off_by_default)
{
    compiled_budgets::get().budgets.clear();
    migraphx::gpu::context ctx;
    migraphx::module m;
    auto x      = m.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
    auto output = m.add_parameter("output", {migraphx::shape::float_type, {2, 3}});
    auto tuned  = m.add_instruction(
        migraphx::make_op("gpu::precompile_op",
                          {{"op", migraphx::to_value(migraphx::make_op("test::tuned"))}}),
        x,
        output);
    m.add_return({tuned});

    EXPECT(test::throws<migraphx::exception>(
        [&] { migraphx::run_passes(m, {migraphx::gpu::compile_ops{.ctx = &ctx}}); },
        "No valid tuned compilation"));
    std::map<std::size_t, migraphx::optional<std::chrono::milliseconds>> expected = {
        {0, migraphx::nullopt}, {1, migraphx::nullopt}, {2, migraphx::nullopt}};
    EXPECT(compiled_budgets::get().budgets == expected);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
