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

#include <migraphx/program.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/register_target.hpp>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>
#include <hip/hip_runtime_api.h>
#include <test.hpp>

// Verifies MIGRAPHX_SHARE_LITERALS on behavior. The env var is set in main()
// so it is in effect before the first compile in this process. The off/default
// case lives in share_literals.cpp, as a separate process: the pool is
// process-scoped, so an off-case assertion here would see this file's residue.
//
// Sharing is detected by the free-VRAM delta after compiling: a large weight
// makes "one copy vs N copies" unmistakable regardless of allocator noise.
//
// The negative cases below are the point of this file. A test suite that only
// compiles identical programs would pass even if finalize shared every literal
// unconditionally without comparing anything, so both directions are asserted.

namespace {
// A weight big enough that one-vs-many copies is obvious (128 MiB of float).
migraphx::shape weight_shape() { return {migraphx::shape::float_type, {32 * 1024 * 1024}}; }

std::size_t free_vram()
{
    std::size_t free_bytes = 0;
    std::size_t total      = 0;
    EXPECT(hipMemGetInfo(&free_bytes, &total) == hipSuccess);
    return free_bytes;
}

// Compiles a program holding the given weight and returns it (kept alive by the
// caller so its device memory stays resident).
migraphx::program make_program_from(const migraphx::target& t, const migraphx::literal& weight)
{
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto w    = mm->add_literal(weight);
    auto relu = mm->add_instruction(migraphx::make_op("relu"), w);
    mm->add_return({relu});
    p.compile(t);
    return p;
}

migraphx::program make_weight_program(const migraphx::target& t)
{
    return make_program_from(t, migraphx::generate_literal(weight_shape(), 0));
}
} // namespace

TEST_CASE(share_when_flag_on)
{
    auto t                = migraphx::make_target("gpu");
    const std::size_t wsz = weight_shape().bytes();

    auto a               = make_weight_program(t);
    const std::size_t f1 = free_vram();
    auto b               = make_weight_program(t); // identical weight, second program
    const std::size_t f2 = free_vram();

    // Flag on: the second identical program aliases the existing device buffer,
    // so free VRAM barely moves (well under a full weight).
    EXPECT((f1 - f2) < (wsz / 2));
}

TEST_CASE(share_across_many_programs)
{
    auto t                = migraphx::make_target("gpu");
    const std::size_t wsz = weight_shape().bytes();

    std::vector<migraphx::program> progs;
    progs.push_back(make_weight_program(t)); // A
    const std::size_t f1 = free_vram();
    progs.push_back(make_weight_program(t)); // B
    progs.push_back(make_weight_program(t)); // C
    progs.push_back(make_weight_program(t)); // D
    const std::size_t f2 = free_vram();

    // B, C and D all alias A's device buffer: three more identical programs add
    // well under one extra weight in total.
    EXPECT((f1 - f2) < (wsz / 2));
}

TEST_CASE(different_weights_do_not_share)
{
    auto t                = migraphx::make_target("gpu");
    const std::size_t wsz = weight_shape().bytes();

    auto a               = make_program_from(t, migraphx::generate_literal(weight_shape(), 0));
    const std::size_t f1 = free_vram();
    auto b               = make_program_from(t, migraphx::generate_literal(weight_shape(), 1));
    const std::size_t f2 = free_vram();

    // Different contents must not alias, so the second program costs a full
    // weight even with the flag on.
    EXPECT((f1 - f2) > (wsz / 2));
}

TEST_CASE(one_ulp_apart_does_not_share)
{
    auto t                = migraphx::make_target("gpu");
    const std::size_t wsz = weight_shape().bytes();

    // Two weights differing in a single element by one ULP. This is the case
    // argument::operator== gets wrong: for floating-point types it compares
    // element-wise with float_equal, a 1-ULP tolerance check, so it would
    // report these equal and silently substitute one weight for the other.
    // The pool confirms candidates with std::memcmp precisely to avoid that.
    auto base = migraphx::generate_literal(weight_shape(), 0);
    std::vector<float> v;
    base.visit([&](auto x) { v.assign(x.begin(), x.end()); });
    auto& e = v.at(v.size() / 2);
    e       = std::nextafter(e, std::numeric_limits<float>::max());

    auto a               = make_program_from(t, base);
    const std::size_t f1 = free_vram();
    auto b               = make_program_from(t, migraphx::literal{weight_shape(), v});
    const std::size_t f2 = free_vram();

    EXPECT((f1 - f2) > (wsz / 2));
}

int main(int argc, const char* argv[])
{
    // Must be set before the first compile in this process: gpu_literal reads it
    // in finalize, and any literal pooled before it is set would stay unpooled.
#ifdef _WIN32
    _putenv_s("MIGRAPHX_SHARE_LITERALS", "1");
#else
    setenv("MIGRAPHX_SHARE_LITERALS", "1", 1);
#endif
    test::run(argc, argv);
}
