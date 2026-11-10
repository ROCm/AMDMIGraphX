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
#include <migraphx/gpu/fuse_gather_dot.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/ranges.hpp>
#include <test.hpp>
#include "kernel_names.hpp"
#include <pointwise.hpp>

static void run_pass(migraphx::program& p, const migraphx::gpu::fuse_gather_dot& pass = {})
{
    migraphx::run_passes(p, {pass, migraphx::dead_code_elimination{}});
}

// The dot of each row with the matrix it gathers reads the table instead
TEST_CASE(gather_dot)
{
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {8, 1, 3}});
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {4, 3, 5}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto wg  = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, idx);
        auto dot = mm->add_instruction(migraphx::make_op("dot"), x, wg);
        mm->add_return({dot});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {8, 1, 3}});
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {4, 3, 5}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto dot = mm->add_instruction(
            migraphx::make_op(
                "gpu::gather_dot",
                {{"layout",
                  migraphx::to_value(migraphx::shape{migraphx::shape::float_type, {4, 3, 5}})}}),
            x,
            idx,
            w);
        mm->add_return({dot});
    }
    EXPECT(p1.sort() == p2.sort());
}

// The transpose of the gathered matrices is applied to the table
TEST_CASE(gather_dot_transpose)
{
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {8, 1, 3}});
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {4, 5, 3}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto wg  = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, idx);
        auto wt =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), wg);
        auto dot = mm->add_instruction(migraphx::make_op("dot"), x, wt);
        mm->add_return({dot});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {8, 1, 3}});
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {4, 5, 3}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto wt =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), w);
        auto dot = mm->add_instruction(
            migraphx::make_op(
                "gpu::gather_dot",
                {{"layout",
                  migraphx::to_value(migraphx::shape{migraphx::shape::float_type, {4, 3, 5}})}}),
            x,
            idx,
            wt);
        mm->add_return({dot});
    }
    EXPECT(p1.sort() == p2.sort());
}

// The pointwise operator computing the gathered matrices and the unpack_int4
// of its input compute the table instead
TEST_CASE(gather_dot_pointwise_unpack_int4)
{
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto x      = mm->add_parameter("x", {migraphx::shape::half_type, {8, 1, 4}});
        auto w      = mm->add_parameter("w", {migraphx::shape::uint8_type, {4, 6, 2}});
        auto scales = mm->add_parameter("scales", {migraphx::shape::half_type, {4, 6}});
        auto idx    = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto wg     = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, idx);
        auto wu     = mm->add_instruction(migraphx::make_op("unpack_int4"), wg);
        auto sg     = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), scales, idx);
        auto su     = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), sg);
        auto sb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {8, 6, 4}}}), su);
        auto dq = add_pointwise(p1, "main:pointwise0", {wu, sb}, [](auto* pm, const auto& inputs) {
            auto q = pm->add_instruction(
                migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
                inputs[0]);
            return pm->add_instruction(migraphx::make_op("mul"), q, inputs[1]);
        });
        auto wt =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), dq);
        auto dot = mm->add_instruction(migraphx::make_op("dot"), x, wt);
        mm->add_return({dot});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm    = p2.get_main_module();
        auto x      = mm->add_parameter("x", {migraphx::shape::half_type, {8, 1, 4}});
        auto w      = mm->add_parameter("w", {migraphx::shape::uint8_type, {4, 6, 2}});
        auto scales = mm->add_parameter("scales", {migraphx::shape::half_type, {4, 6}});
        auto idx    = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto wu     = mm->add_instruction(migraphx::make_op("unpack_int4"), w);
        auto su     = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), scales);
        auto sb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 6, 4}}}), su);
        auto dq = add_pointwise(p2, "main:pointwise0", {wu, sb}, [](auto* pm, const auto& inputs) {
            auto q = pm->add_instruction(
                migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
                inputs[0]);
            return pm->add_instruction(migraphx::make_op("mul"), q, inputs[1]);
        });
        auto wt =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), dq);
        auto dot = mm->add_instruction(
            migraphx::make_op(
                "gpu::gather_dot",
                {{"layout",
                  migraphx::to_value(migraphx::shape{migraphx::shape::half_type, {4, 4, 6}})}}),
            x,
            idx,
            wt);
        mm->add_return({dot});
    }
    EXPECT(p1.sort() == p2.sort());
}

// An input of the pointwise operator that is one row broadcast to all the
// rows is broadcast to the entries of the table
TEST_CASE(gather_dot_pointwise_row_broadcast)
{
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {8, 1, 3}});
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {4, 3, 5}});
        auto s   = mm->add_parameter("s", {migraphx::shape::float_type, {3, 5}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto wg  = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, idx);
        auto sb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {8, 3, 5}}}), s);
        auto ws  = add_pointwise(p1, "main:pointwise0", {wg, sb}, single_pointwise("mul"));
        auto dot = mm->add_instruction(migraphx::make_op("dot"), x, ws);
        mm->add_return({dot});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {8, 1, 3}});
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {4, 3, 5}});
        auto s   = mm->add_parameter("s", {migraphx::shape::float_type, {3, 5}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto sb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 3, 5}}}), s);
        auto ws  = add_pointwise(p2, "main:pointwise0", {w, sb}, single_pointwise("mul"));
        auto dot = mm->add_instruction(
            migraphx::make_op(
                "gpu::gather_dot",
                {{"layout",
                  migraphx::to_value(migraphx::shape{migraphx::shape::float_type, {4, 3, 5}})}}),
            x,
            idx,
            ws);
        mm->add_return({dot});
    }
    EXPECT(p1.sort() == p2.sort());
}

// With too few rows for the table the dots keep their gathers: 3 rows from 8
// entries
TEST_CASE(gather_dot_few_rows)
{
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {3, 1, 3}});
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {8, 3, 5}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {3}});
        auto wg  = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, idx);
        auto dot = mm->add_instruction(migraphx::make_op("dot"), x, wg);
        mm->add_return({dot});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1 == p2);
}

// A lower threshold groups them: 3 rows from 8 entries is 0.375 rows per entry
TEST_CASE(gather_dot_few_rows_lower_threshold)
{
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {3, 1, 3}});
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {8, 3, 5}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {3}});
        auto wg  = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, idx);
        auto dot = mm->add_instruction(migraphx::make_op("dot"), x, wg);
        mm->add_return({dot});
    }
    run_pass(p1, {.min_rows_per_entry = 0.25f});

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {3, 1, 3}});
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {8, 3, 5}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {3}});
        auto dot = mm->add_instruction(
            migraphx::make_op(
                "gpu::gather_dot",
                {{"layout",
                  migraphx::to_value(migraphx::shape{migraphx::shape::float_type, {8, 3, 5}})}}),
            x,
            idx,
            w);
        mm->add_return({dot});
    }
    EXPECT(p1.sort() == p2.sort());
}

// Matrices gathered with other indices than the scales are not computed
// from one entry
TEST_CASE(gather_dot_different_indices)
{
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto x      = mm->add_parameter("x", {migraphx::shape::float_type, {8, 1, 3}});
        auto w      = mm->add_parameter("w", {migraphx::shape::float_type, {4, 3, 5}});
        auto scales = mm->add_parameter("scales", {migraphx::shape::float_type, {4, 3, 5}});
        auto idx1   = mm->add_parameter("idx1", {migraphx::shape::int32_type, {8}});
        auto idx2   = mm->add_parameter("idx2", {migraphx::shape::int32_type, {8}});
        auto wg     = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, idx1);
        auto sg     = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), scales, idx2);
        auto ws     = mm->add_instruction(migraphx::make_op("mul"), wg, sg);
        auto dot    = mm->add_instruction(migraphx::make_op("dot"), x, ws);
        mm->add_return({dot});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1 == p2);
}

// A dot with more than one row per matrix is left to the other dot kernels
TEST_CASE(gather_dot_several_rows)
{
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {8, 2, 3}});
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {4, 3, 5}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto wg  = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, idx);
        auto dot = mm->add_instruction(migraphx::make_op("dot"), x, wg);
        mm->add_return({dot});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1 == p2);
}

// The whole pipeline compiles the gathered dot of packed weights to the
// grouped kernel, with the gathers and the dequantize inside it
TEST_CASE(compile_gather_dot_unpack_int4)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto x      = mm->add_parameter("x", {migraphx::shape::half_type, {8, 1, 32}});
    auto w      = mm->add_parameter("w", {migraphx::shape::uint8_type, {4, 16, 16}});
    auto scales = mm->add_parameter("scales", {migraphx::shape::float_type, {4, 16}});
    auto idx    = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
    auto zero   = mm->add_literal(migraphx::literal{{migraphx::shape::uint8_type, {1}}, {8}});
    auto wg     = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, idx);
    auto wu     = mm->add_instruction(migraphx::make_op("unpack_int4"), wg);
    auto sg     = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), scales, idx);
    auto su     = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), sg);
    auto sb =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {8, 16, 32}}}), su);
    auto zb =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {8, 16, 32}}}), zero);
    auto dq = mm->add_instruction(migraphx::make_op("dequantizelinear"), wu, sb, zb);
    auto wh = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), dq);
    auto wt = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), wh);
    auto dot = mm->add_instruction(migraphx::make_op("dot"), x, wt);
    mm->add_return({dot});
    p.compile(migraphx::make_target("gpu"));

    auto kernels = get_kernel_names(p);
    EXPECT(kernels == std::vector<std::string>{"gather_dot_kernel"});
}

// The add of the gathered bias and the pointwise operator over pairs of
// columns are computed in the grouped kernel too, which leaves the gather of
// the bias
TEST_CASE(compile_gather_dot_bias_pairs)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto x     = mm->add_parameter("x", {migraphx::shape::half_type, {8, 1, 32}});
    auto w     = mm->add_parameter("w", {migraphx::shape::half_type, {4, 32, 16}});
    auto bias  = mm->add_parameter("bias", {migraphx::shape::half_type, {4, 16}});
    auto idx   = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
    auto wg    = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, idx);
    auto dot   = mm->add_instruction(migraphx::make_op("dot"), x, wg);
    auto bg    = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), bias, idx);
    auto bu    = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), bg);
    auto add   = mm->add_instruction(migraphx::make_op("add"), dot, bu);
    auto pairs = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {8, 1, 8, 2}}}), add);
    auto gate  = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {1}}}), pairs);
    auto up = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {3}}, {"starts", {1}}, {"ends", {2}}}), pairs);
    auto sig = mm->add_instruction(migraphx::make_op("sigmoid"), gate);
    auto mul = mm->add_instruction(migraphx::make_op("mul"), sig, up);
    mm->add_return({mul});
    p.compile(migraphx::make_target("gpu"));

    auto kernels = get_kernel_names(p);
    EXPECT(kernels == std::vector<std::string>{"gather_kernel", "gather_dot_kernel"});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
