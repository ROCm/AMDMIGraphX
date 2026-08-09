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
#include <migraphx/gpu/fuse_skinny_gemm.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/generate.hpp>
#include <pointwise.hpp>
#include <test.hpp>

static void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::gpu::fuse_skinny_gemm{}, migraphx::dead_code_elimination{}});
}

static bool contains_op(const migraphx::module& m, const std::string& name)
{
    return std::any_of(m.begin(), m.end(), [&](const auto& ins) { return ins.name() == name; });
}

static migraphx::program
make_dot_add(std::size_t m_dim, std::size_t k, std::size_t n, bool with_add)
{
    migraphx::shape a_s{migraphx::shape::bf16_type, {m_dim, 1, k}};
    migraphx::shape w_s{migraphx::shape::bf16_type, {k, n}};
    migraphx::shape o_s{migraphx::shape::bf16_type, {m_dim, 1, n}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto a   = mm->add_parameter("a", a_s);
    auto w   = mm->add_literal(migraphx::generate_literal(w_s, 0));
    auto wb =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {m_dim, k, n}}}), w);
    auto dot = mm->add_instruction(migraphx::make_op("dot"), a, wb);
    if(with_add)
    {
        auto res = mm->add_parameter("res", o_s);
        dot      = add_pointwise(p, "main:pointwise0", {res, dot}, single_pointwise("add"));
    }
    mm->add_return({dot});
    return p;
}

TEST_CASE(fuse_skinny_gemm_with_residual)
{
    auto p = make_dot_add(4, 4096, 4096, true);
    run_pass(p);
    const auto* mm = p.get_main_module();
    EXPECT(contains_op(*mm, "gpu::skinny_gemm_splitk"));
    EXPECT(contains_op(*mm, "gpu::skinny_gemm_reduce"));
    EXPECT(not contains_op(*mm, "dot"));
    auto reduce = std::find_if(mm->begin(), mm->end(), [](const auto& ins) {
        return ins.name() == "gpu::skinny_gemm_reduce";
    });
    EXPECT(reduce->inputs().size() == 2);
    EXPECT(reduce->get_shape().lens() == std::vector<std::size_t>{4, 1, 4096});
}

// Without a residual to fold the reduce kernel is pure overhead: keep the dot
TEST_CASE(fuse_skinny_gemm_requires_residual)
{
    auto p = make_dot_add(4, 4096, 4096, false);
    run_pass(p);
    EXPECT(not contains_op(*p.get_main_module(), "gpu::skinny_gemm_splitk"));
    EXPECT(contains_op(*p.get_main_module(), "dot"));
}

// Too many rows: the weight is no longer the dominant traffic
TEST_CASE(fuse_skinny_gemm_skips_wide_m)
{
    auto p = make_dot_add(64, 4096, 4096, true);
    run_pass(p);
    EXPECT(not contains_op(*p.get_main_module(), "gpu::skinny_gemm_splitk"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
