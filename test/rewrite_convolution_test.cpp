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
#include <migraphx/rewrite_convolution.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::rewrite_convolution{}, migraphx::dead_code_elimination{}});
}

static migraphx::shape sf(std::vector<std::size_t> lens)
{
    return migraphx::shape{migraphx::shape::float_type, std::move(lens)};
}

// stride 1: a backward-data convolution is one flipped stride-1 forward convolution.
TEST_CASE(stride1_single_forward_conv)
{
    migraphx::module m1;
    {
        auto dy = m1.add_parameter("dy", sf({1, 2, 5}));
        auto w  = m1.add_parameter("w", sf({2, 3, 3}));
        auto r  = m1.add_instruction(
            migraphx::make_op("convolution_backwards",
                               {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}, {"group", 1}}),
            dy,
            w);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto dy = m2.add_parameter("dy", sf({1, 2, 5}));
        auto w  = m2.add_parameter("w", sf({2, 3, 3}));
        auto t =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), w);
        auto rv = m2.add_instruction(migraphx::make_op("reverse", {{"axes", {2}}}), t);
        auto c  = m2.add_instruction(
            migraphx::make_op(
                "convolution",
                {{"padding", {2, 2}}, {"stride", {1}}, {"dilation", {1}}, {"group", 1}}),
            dy,
            rv);
        m2.add_return({c});
    }
    EXPECT(m1 == m2);
}

// stride 2: two residues -> two stride-1 forward convolutions interleaved back together.
TEST_CASE(stride2_two_residue_interleave)
{
    migraphx::module m1;
    {
        auto dy = m1.add_parameter("dy", sf({1, 1, 3}));
        auto w  = m1.add_parameter("w", sf({1, 1, 2}));
        auto r  = m1.add_instruction(
            migraphx::make_op("convolution_backwards",
                               {{"padding", {0}}, {"stride", {2}}, {"dilation", {1}}, {"group", 1}}),
            dy,
            w);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto dy = m2.add_parameter("dy", sf({1, 1, 3}));
        auto w  = m2.add_parameter("w", sf({1, 1, 2}));

        auto conv_residue = [&](migraphx::instruction_ref wslice, int64_t before, int64_t after) {
            auto st = m2.add_instruction(migraphx::make_op("step", {{"axes", {2}}, {"steps", {2}}}),
                                         wslice);
            auto t  = m2.add_instruction(
                migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), st);
            auto rv = m2.add_instruction(migraphx::make_op("reverse", {{"axes", {2}}}), t);
            auto c  = m2.add_instruction(
                migraphx::make_op(
                    "convolution",
                    {{"padding", {0, 0}}, {"stride", {1}}, {"dilation", {1}}, {"group", 1}}),
                dy,
                rv);
            auto u = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3}}}), c);
            auto p = m2.add_instruction(
                migraphx::make_op("pad", {{"pads", {0, 0, 0, 0, 0, 0, 0, 1}}}), u);
            auto rs = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 6}}}), p);
            return m2.add_instruction(
                migraphx::make_op("pad", {{"pads", {0, 0, before, 0, 0, after}}}), rs);
        };

        // residue 0 (itilda=0): no leading slice (full axis); placed at offset 0
        auto r0 = conv_residue(w, 0, 1);
        // residue 1 (itilda=1): slice taps starting at 1; placed at offset 1
        auto s1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {2}}}), w);
        auto r1  = conv_residue(s1, 1, 0);
        auto add = m2.add_instruction(migraphx::make_op("add"), r0, r1);
        auto cr  = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {6}}}), add);
        m2.add_return({cr});
    }
    EXPECT(m1 == m2);
}

// grouped backward-data convolution: the in/out channel swap keeps groups intact via reshape.
TEST_CASE(grouped_weight_reshape)
{
    migraphx::module m1;
    {
        auto dy = m1.add_parameter("dy", sf({1, 4, 5}));
        auto w  = m1.add_parameter("w", sf({4, 2, 3}));
        auto r  = m1.add_instruction(
            migraphx::make_op("convolution_backwards",
                               {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}, {"group", 2}}),
            dy,
            w);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto dy    = m2.add_parameter("dy", sf({1, 4, 5}));
        auto w     = m2.add_parameter("w", sf({4, 2, 3}));
        auto split = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 3}}}), w);
        auto trans = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), split);
        auto merge = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 2, 3}}}), trans);
        auto rv    = m2.add_instruction(migraphx::make_op("reverse", {{"axes", {2}}}), merge);
        auto c     = m2.add_instruction(
            migraphx::make_op(
                "convolution",
                {{"padding", {2, 2}}, {"stride", {1}}, {"dilation", {1}}, {"group", 2}}),
            dy,
            rv);
        m2.add_return({c});
    }
    EXPECT(m1 == m2);
}

// dynamic shapes are left untouched (existing path handles them).
TEST_CASE(dynamic_shape_no_rewrite)
{
    auto make = [] {
        migraphx::module m;
        migraphx::shape dys{migraphx::shape::float_type, {{1, 1}, {1, 1}, {3, 8}}};
        auto dy = m.add_parameter("dy", dys);
        auto w  = m.add_parameter("w", sf({1, 1, 2}));
        auto r  = m.add_instruction(
            migraphx::make_op("convolution_backwards",
                               {{"padding", {0}}, {"stride", {2}}, {"dilation", {1}}, {"group", 1}}),
            dy,
            w);
        m.add_return({r});
        return m;
    };
    migraphx::module m1 = make();
    run_pass(m1);
    migraphx::module m2 = make();
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
