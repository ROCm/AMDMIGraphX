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
#include <migraphx/register_target.hpp>
#include <vector>
#include <hip/hip_runtime_api.h>
#include <test.hpp>

// Verifies MIGRAPHX_SHARE_LITERALS default (off) behavior: two programs with
// identical weights each upload their own device copy, so weight VRAM is NOT
// shared. The companion file share_literals_on.cpp checks the flag-on case.
// They are separate processes because the pool is process-scoped: once a
// literal is pooled it stays pooled, so an off-case assertion in the same
// process as an on-case would see the on-case's residue.
//
// Sharing is detected by the free-VRAM delta after compiling: a large weight
// makes "one copy vs N copies" unmistakable regardless of allocator noise.

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

// Compiles a program holding one copy of the same weight and returns it (kept
// alive by the caller so its device memory stays resident).
migraphx::program make_weight_program(const migraphx::target& t)
{
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto w    = mm->add_literal(migraphx::generate_literal(weight_shape(), 0));
    auto relu = mm->add_instruction(migraphx::make_op("relu"), w);
    mm->add_return({relu});
    p.compile(t);
    return p;
}
} // namespace

TEST_CASE(no_share_when_flag_off)
{
    auto t                = migraphx::make_target("gpu");
    const std::size_t wsz = weight_shape().bytes();

    auto a               = make_weight_program(t);
    const std::size_t f1 = free_vram();
    auto b               = make_weight_program(t); // identical weight, second program
    const std::size_t f2 = free_vram();

    // Flag off: the second identical program uploads its own copy, so free VRAM
    // drops by roughly another full weight.
    EXPECT((f1 - f2) > (wsz / 2));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
