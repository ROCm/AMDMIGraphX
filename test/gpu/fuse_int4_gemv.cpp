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

#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/gpu/fuse_int4_gemv.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <test.hpp>

#include <cstdlib>

// The fusion is opt-in behind MIGRAPHX_ENABLE_INT4_GEMV, and migraphx::enabled() memoises the
// lookup in a function-local `static const bool` -- so the value is frozen at the first query in
// the process and a setenv() inside a TEST_CASE body would be too late to have any effect.
//
// Each file under test/gpu is built into its own executable (see test/CMakeLists.txt), so this
// process is ours alone: setting the variable from a file-scope initialiser runs it before main,
// and therefore before anything has had a chance to query and memoise it. Without this the pass
// would early-return, every expectation below would compare "unfused vs unfused", and the whole
// file would pass green while testing nothing.
namespace {
struct enable_int4_gemv_env
{
    enable_int4_gemv_env()
    {
#ifdef _WIN32
        _putenv_s("MIGRAPHX_ENABLE_INT4_GEMV", "1");
#else
        setenv("MIGRAPHX_ENABLE_INT4_GEMV", "1", 1);
#endif
    }
};
const enable_int4_gemv_env enable_env{};

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(
        p, {migraphx::gpu::fuse_int4_gemv{}, migraphx::dead_code_elimination{}});
}

std::size_t count_op(const migraphx::program& p, const std::string& name)
{
    const auto* mm = p.get_main_module();
    return std::count_if(
        mm->begin(), mm->end(), [&](const auto& ins) { return ins.name() == name; });
}

// A decode-shaped INT4 weight-only GEMV:  A{batch,1,K} x dequant(unpack_int4(W)){K,N}
// K and N default to Llama-3.2-1B's fused gate_up (16384x2048), the shape that dominates the
// decode step in practice. `zp` selects the asymmetric form -- symmetry is the axis the matcher
// branches on, so both are exercised below.
migraphx::program make_int4_gemv_program(std::size_t k = 2048,
                                         std::size_t n = 16384,
                                         std::size_t batch = 1,
                                         std::size_t m     = 1,
                                         bool zp           = true)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    // int4 weights arrive packed two-per-byte, so the stored K is half the logical K.
    migraphx::shape a_shape{migraphx::shape::half_type, {batch, m, k}};
    migraphx::shape w_shape{migraphx::shape::uint8_type, {k / 2, n}};
    migraphx::shape s_shape{migraphx::shape::half_type, {k / 32, n}};

    auto a      = mm->add_parameter("a", a_shape);
    auto w      = mm->add_parameter("w", w_shape);
    auto scale  = mm->add_parameter("scale", s_shape);
    auto unpack = mm->add_instruction(migraphx::make_op("unpack_int4"), w);

    migraphx::instruction_ref dq;
    if(zp)
    {
        auto zeros = mm->add_parameter("zp", {migraphx::shape::uint8_type, {k / 32, n}});
        dq = mm->add_instruction(migraphx::make_op("dequantizelinear"), unpack, scale, zeros);
    }
    else
    {
        dq = mm->add_instruction(migraphx::make_op("dequantizelinear"), unpack, scale);
    }

    auto dot = mm->add_instruction(migraphx::make_op("dot"), a, dq);
    mm->add_return({dot});
    return p;
}
} // namespace

TEST_CASE(fuses_m1_int4_gemv)
{
    auto p = make_int4_gemv_program();
    run_pass(p);
    // The dot is replaced, not merely annotated -- assert both directions so a pass that inserts
    // the fused op while leaving the original dot behind is still a failure.
    EXPECT(count_op(p, "gpu::int4_gemv") == 1);
    EXPECT(count_op(p, "dot") == 0);
}

TEST_CASE(fuses_symmetric_int4_gemv)
{
    auto p = make_int4_gemv_program(2048, 16384, 1, 1, false);
    run_pass(p);
    EXPECT(count_op(p, "gpu::int4_gemv") == 1);
    EXPECT(count_op(p, "dot") == 0);
}

// --- negative cases -------------------------------------------------------------------------
// These exist so the assertions above can actually fail. A matcher that fired unconditionally
// would satisfy every positive case and still be wrong; each test below is a shape the kernel
// does not implement, and must be left for the default codegen path.

TEST_CASE(skips_when_m_is_not_one)
{
    auto p = make_int4_gemv_program(2048, 16384, 1, 8);
    run_pass(p);
    EXPECT(count_op(p, "gpu::int4_gemv") == 0);
    EXPECT(count_op(p, "dot") == 1);
}

TEST_CASE(skips_when_batch_makes_gm_exceed_one)
{
    // The gate is on the product of the batch dims and M, not on M alone: {4,1,K} is four
    // independent GEMVs, which this kernel does not handle.
    auto p = make_int4_gemv_program(2048, 16384, 4, 1);
    run_pass(p);
    EXPECT(count_op(p, "gpu::int4_gemv") == 0);
    EXPECT(count_op(p, "dot") == 1);
}

TEST_CASE(skips_plain_half_dot)
{
    // No unpack_int4/dequantizelinear in the B chain -- an ordinary fp16 GEMV must be untouched.
    migraphx::program p;
    {
        auto* mm = p.get_main_module();
        auto a   = mm->add_parameter("a", {migraphx::shape::half_type, {1, 1, 2048}});
        auto b   = mm->add_parameter("b", {migraphx::shape::half_type, {2048, 16384}});
        auto dot = mm->add_instruction(migraphx::make_op("dot"), a, b);
        mm->add_return({dot});
    }
    run_pass(p);
    EXPECT(count_op(p, "gpu::int4_gemv") == 0);
    EXPECT(count_op(p, "dot") == 1);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
