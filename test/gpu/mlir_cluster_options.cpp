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
// Option validation for mlir_num_ctas / mlir_warp_specialize, modeled on lds_carveout.cpp.
// MIGraphX only validates the option's own value range/enum and forwards it unconditionally;
// whether a device can actually honor a non-default request is decided by rocMLIR during
// lowering, not by MIGraphX, so these tests only assert that in-range values never throw at the
// MIGraphX layer. A device-specific test of the actual cluster/warp-specialize rewrite is in
// mlir_perf_config.cpp (pure logic).
#include <migraphx/program.hpp>
#include <migraphx/compile_modes.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <test.hpp>

namespace {

migraphx::program compile_conv(const migraphx::value& num_ctas,
                               const migraphx::value& warp_spec,
                               bool eager = false)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {1, 8, 8, 8}});
    auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {8, 8, 3, 3}});
    mm->add_return({mm->add_instruction(migraphx::make_op("convolution"), x, w)});

    migraphx::compile_options options;
    if(not num_ctas.is_null())
        options.backend_options["mlir_num_ctas"] = num_ctas;
    if(not warp_spec.is_null())
        options.backend_options["mlir_warp_specialize"] = warp_spec;
    // eager mode compiles only the first candidate and skips benchmarking, so it exercises
    // code generation (and option validation) without actually launching a kernel.
    if(eager)
        options.compile_mode = migraphx::compile_modes::eager;
    p.compile(migraphx::make_target("gpu"), options);
    return p;
}

} // namespace

TEST_CASE(mlir_num_ctas_default_compiles)
{
    EXPECT(compile_conv(migraphx::value{}, migraphx::value{}).size() > 0);
}

// Every in-range value is accepted and forwarded; MIGraphX does not throw based on the device.
// rocMLIR decides whether the device can actually honor a non-default cluster size.
TEST_CASE(mlir_num_ctas_valid_values_compile)
{
    for(int v : {0, 1, 2, 4, 8})
        EXPECT(compile_conv(v, migraphx::value{}).size() > 0);
}

// 16 is a legal option value. Eager mode compiles only the first candidate and skips benchmarking,
// so this exercises option validation and rocMLIR code generation without launching a kernel.
// rocMLIR can reject a clustered candidate on a device with a lower cluster limit (gfx1151 allows
// one CTA per cluster). This case passes because the unclustered fallback is the one compiled.
TEST_CASE(mlir_num_ctas_sixteen_is_a_valid_option)
{
    EXPECT(compile_conv(16, migraphx::value{}, /*eager=*/true).size() > 0);
}

TEST_CASE(mlir_num_ctas_invalid_values_throw)
{
    EXPECT(test::throws([] { compile_conv(3, migraphx::value{}); }));
    EXPECT(test::throws([] { compile_conv(32, migraphx::value{}); }));
    EXPECT(test::throws([] { compile_conv(-1, migraphx::value{}); }));
}

// Every valid mode is accepted and forwarded; MIGraphX does not throw based on the device.
TEST_CASE(mlir_warp_specialize_valid_values_compile)
{
    for(const auto& v : {"off", "tune", "force"})
        EXPECT(compile_conv(migraphx::value{}, std::string{v}).size() > 0);
}

TEST_CASE(mlir_warp_specialize_invalid_value_throws)
{
    EXPECT(test::throws([] { compile_conv(migraphx::value{}, std::string{"always"}); }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
