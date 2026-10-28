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
// disable_hiprtc_processes cannot be undone, so these tests live in their own executable where
// it cannot change how the other GPU tests compile. They run in the order they are declared.
#include <test.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/value.hpp>
#include <migraphx/gpu/kernel.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/gpu/compile_hip.hpp>

// NOLINTNEXTLINE
const std::string write_2s = R"__migraphx__(
#ifndef __HIPCC_RTC__
#include <hip/hip_runtime.h>
#endif

extern "C" {
__global__ void write(char* data)
{
    int num = threadIdx.x + blockDim.x * blockIdx.x;
    data[num] = 2;
}

}

int main() {}

)__migraphx__";

// relu lowers to gpu::precompile_op, so compile_ops compiles it with hiprtc.
TEST_CASE(backend_option_disables_processes)
{
    EXPECT(not migraphx::gpu::hiprtc_processes_disabled());

    migraphx::program p_ref;
    auto* mm = p_ref.get_main_module();
    mm->add_return({mm->add_instruction(
        migraphx::make_op("relu"),
        mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {5, 2}})))});
    migraphx::program p_gpu = p_ref;

    p_ref.compile(migraphx::make_target("ref"), migraphx::compile_options{});
    auto expected = p_ref.eval({}).front();

    migraphx::compile_options options;
    options.backend_options["hiprtc_disable_processes"] = migraphx::value(true);
    p_gpu.compile(migraphx::make_target("gpu"), options);
    EXPECT(migraphx::gpu::hiprtc_processes_disabled());

    auto result = migraphx::gpu::from_gpu(p_gpu.eval({}).front());
    EXPECT(result == expected);
}

TEST_CASE(compile_hip_src_in_process)
{
    migraphx::gpu::disable_hiprtc_processes();
    auto binaries = migraphx::gpu::compile_hip_src(
        {migraphx::src_file{"main.cpp", write_2s}}, {}, migraphx::gpu::get_device_name());
    EXPECT(binaries.size() == 1);

    migraphx::argument input{{migraphx::shape::int8_type, {5}}};
    auto ginput = migraphx::gpu::to_gpu(input);
    migraphx::gpu::kernel k{binaries.front(), "write"};
    k.launch(nullptr, input.get_shape().elements(), 1024)(ginput.cast<std::int8_t>());
    auto output = migraphx::gpu::from_gpu(ginput);

    EXPECT(output != input);
    auto data = output.get<std::int8_t>();
    EXPECT(migraphx::all_of(data, [](auto x) { return x == 2; }));
}

// The compiler probes go through compile_hip_src too, and must give the same answers in-process
// as they do through migraphx-hiprtc-driver (see has_flags in jit.cpp).
TEST_CASE(probes_in_process)
{
    migraphx::gpu::disable_hiprtc_processes();
    EXPECT(migraphx::gpu::hip_has_flags({"--std=c++17"}));
    EXPECT(not migraphx::gpu::hip_has_flags({"--non-existent-flag-to-test-in-migraphx"}));
    EXPECT(migraphx::gpu::hip_has_flags({"-Wunused-parameter"}));
    EXPECT(not migraphx::gpu::hip_has_flags(
        {"-Wnon-existent-warnings-flag-to-test-in-migraphx", "-Werror"}));
    EXPECT(not migraphx::gpu::hip_compiler_version().empty());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
