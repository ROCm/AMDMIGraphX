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
 *
 */
#include <migraphx/gpu/mlir.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/stringutils.hpp>
#include <numeric>
#include <test.hpp>

// The binary cache addresses MLIR kernels by mlir_compile_key, so the key for a module must not
// depend on what the process has compiled before it. rocMLIR sets global printer options when it
// is registered and its linker resets them during the first compile, so this test only means
// something while no MLIR kernel has been compiled yet. It must stay the first test in this
// file, and this file must stay its own executable.
TEST_CASE(mlir_compile_key_is_stable_across_compiles)
{
    migraphx::module m;
    auto arg0 = m.add_parameter("arg0", {migraphx::shape::half_type, {1, 64, 32}});
    auto arg1 = m.add_parameter("arg1", {migraphx::shape::half_type, {1, 32, 48}});
    auto arg2 = m.add_parameter("arg2", {migraphx::shape::half_type, {1, 64, 48}});
    auto dot  = m.add_instruction(migraphx::make_op("dot"), arg0, arg1);
    auto add  = m.add_instruction(migraphx::make_op("add"), dot, arg2);
    m.add_return({add});
    std::vector<migraphx::shape> shapes = {{migraphx::shape::half_type, {1, 64, 32}},
                                           {migraphx::shape::half_type, {1, 32, 48}},
                                           {migraphx::shape::half_type, {1, 64, 48}},
                                           {migraphx::shape::half_type, {1, 64, 48}}};

    migraphx::gpu::context ctx;
    auto tc = migraphx::gpu::get_tuning_config_mlir(ctx, m, shapes, false);
    // Skip when MLIR is not enabled
    if(tc.solutions.empty())
        return;
    const auto& solution = tc.solutions.front();

    auto before = migraphx::gpu::mlir_compile_key(ctx, m, shapes, solution);
    // Skip when MLIR is not enabled
    if(before.empty())
        return;
    auto between = migraphx::gpu::mlir_compile_key(ctx, m, shapes, solution);
    migraphx::gpu::compile_mlir(ctx, m, shapes, solution);
    auto after = migraphx::gpu::mlir_compile_key(ctx, m, shapes, solution);

    EXPECT(before == between);
    EXPECT(before == after);
    // The key is printed in local scope, the form rocMLIR selects before any compile, so keys
    // stored by earlier builds still match. A top-level print would end with a newline.
    EXPECT(not migraphx::ends_with(after, "\n"));
    // Locations never reach the key
    EXPECT(after.find("loc(") == std::string::npos);
}

// Kernels that differ only in the value of a literal must not share a key, however large the
// literal is, since the printer can elide large constants.
TEST_CASE(mlir_compile_key_distinguishes_literals)
{
    migraphx::shape ws{migraphx::shape::float_type, {1, 64, 32}};
    std::vector<float> weights(ws.elements());
    std::iota(weights.begin(), weights.end(), 0.0f);

    migraphx::module m1;
    {
        auto arg0 = m1.add_parameter("arg0", {migraphx::shape::float_type, {1, 16, 64}});
        auto lit  = m1.add_literal(migraphx::literal{ws, weights});
        auto dot  = m1.add_instruction(migraphx::make_op("dot"), arg0, lit);
        m1.add_return({dot});
    }
    weights.back() += 1.0f;
    migraphx::module m2;
    {
        auto arg0 = m2.add_parameter("arg0", {migraphx::shape::float_type, {1, 16, 64}});
        auto lit  = m2.add_literal(migraphx::literal{ws, weights});
        auto dot  = m2.add_instruction(migraphx::make_op("dot"), arg0, lit);
        m2.add_return({dot});
    }
    std::vector<migraphx::shape> shapes = {{migraphx::shape::float_type, {1, 16, 64}},
                                           {migraphx::shape::float_type, {1, 16, 32}}};

    migraphx::gpu::context ctx;
    auto key1 = migraphx::gpu::mlir_compile_key(ctx, m1, shapes, migraphx::value{});
    // Skip when MLIR is not enabled
    if(key1.empty())
        return;
    auto key2 = migraphx::gpu::mlir_compile_key(ctx, m2, shapes, migraphx::value{});
    EXPECT(key1 != key2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
