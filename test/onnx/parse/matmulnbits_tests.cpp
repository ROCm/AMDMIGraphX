/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "migraphx/make_op.hpp"
#include <onnx_test.hpp>

TEST_CASE(matmulnbits_mm_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto a      = mm->add_parameter("a", migraphx::shape{migraphx::shape::float_type, {2, 16}});
    auto b      = mm->add_parameter("b", migraphx::shape{migraphx::shape::uint8_type, {4, 1, 8}});
    auto scales = mm->add_parameter("scales", migraphx::shape{migraphx::shape::float_type, {4}});
    auto zp     = mm->add_parameter("zp", migraphx::shape{migraphx::shape::uint8_type, {4}});

    scales = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, -1}}}), scales);
    scales = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), scales);
    scales = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 1, 16}}}),
                                 scales);
    scales = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, -1}}}), scales);

    zp = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, -1}}}), zp);
    zp = mm->add_instruction(migraphx::make_op("unpack_int4"), zp);
    zp = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), zp);
    zp = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), zp);
    zp = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 1, 16}}}), zp);
    zp = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, -1}}}), zp);

    b = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, -1}}}), b);
    b = mm->add_instruction(migraphx::make_op("unpack_int4"), b);
    b = mm->add_instruction(migraphx::make_op("dequantizelinear"), b, scales, zp);
    b = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), b);
    mm->add_instruction(migraphx::make_op("dot"), a, b);

    auto prog = optimize_onnx("matmulnbits_mm_test.onnx");

    p.sort();
    prog.sort();
    EXPECT(p == prog);
}

TEST_CASE(matmulnbits_mm2_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto a      = mm->add_parameter("a", migraphx::shape{migraphx::shape::float_type, {2, 33}});
    auto b      = mm->add_parameter("b", migraphx::shape{migraphx::shape::uint8_type, {2, 3, 8}});
    auto scales = mm->add_parameter("scales", migraphx::shape{migraphx::shape::float_type, {6}});

    scales = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1}}}), scales);
    scales = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), scales);
    scales = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 16}}}),
                                 scales);
    scales = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1}}}), scales);
    scales = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {33}}}), scales);

    auto zp =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::uint8_type, {1}}, {8}});
    zp = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 33}}}), zp);

    b = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1}}}), b);
    b = mm->add_instruction(migraphx::make_op("unpack_int4"), b);
    b = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {33}}}), b);
    b = mm->add_instruction(migraphx::make_op("dequantizelinear"), b, scales, zp);
    b = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), b);
    mm->add_instruction(migraphx::make_op("dot"), a, b);

    auto prog = optimize_onnx("matmulnbits_mm2_test.onnx");

    p.sort();
    prog.sort();
    EXPECT(p == prog);
}

TEST_CASE(matmulnbits_vm_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto a      = mm->add_parameter("a", migraphx::shape{migraphx::shape::float_type, {20}});
    auto b      = mm->add_parameter("b", migraphx::shape{migraphx::shape::uint8_type, {3, 2, 8}});
    auto scales = mm->add_parameter("scales", migraphx::shape{migraphx::shape::float_type, {6}});
    auto zp     = mm->add_parameter("zp", migraphx::shape{migraphx::shape::uint8_type, {3}});

    scales = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {3, -1}}}), scales);
    scales = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), scales);
    scales = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 2, 16}}}),
                                 scales);
    scales = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {3, -1}}}), scales);
    scales = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {20}}}), scales);

    zp = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {3, -1}}}), zp);
    zp = mm->add_instruction(migraphx::make_op("unpack_int4"), zp);
    zp = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), zp);
    zp = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 2, 16}}}), zp);
    zp = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {3, -1}}}), zp);
    zp = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {20}}}), zp);

    b = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {3, -1}}}), b);
    b = mm->add_instruction(migraphx::make_op("unpack_int4"), b);
    b = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {20}}}), b);
    b = mm->add_instruction(migraphx::make_op("dequantizelinear"), b, scales, zp);
    b = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), b);

    a        = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), a);
    auto dot = mm->add_instruction(migraphx::make_op("dot"), a, b);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), dot);

    auto prog = optimize_onnx("matmulnbits_vm_test.onnx");

    p.sort();
    prog.sort();
    EXPECT(p == prog);
}

TEST_CASE(matmulnbits_bmm_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto a      = mm->add_parameter("a", migraphx::shape{migraphx::shape::float_type, {2, 3, 8}});
    auto b      = mm->add_parameter("b", migraphx::shape{migraphx::shape::uint8_type, {2, 1, 8}});
    auto scales = mm->add_parameter("scales", migraphx::shape{migraphx::shape::float_type, {2}});

    scales = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1}}}), scales);
    scales = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), scales);
    scales = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 1, 16}}}),
                                 scales);
    scales = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1}}}), scales);
    scales = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {8}}}), scales);

    auto zp =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::uint8_type, {1}}, {8}});
    zp = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 8}}}), zp);

    b = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1}}}), b);
    b = mm->add_instruction(migraphx::make_op("unpack_int4"), b);
    b = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {8}}}), b);
    b = mm->add_instruction(migraphx::make_op("dequantizelinear"), b, scales, zp);
    b = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), b);
    b = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 8, 2}}}), b);
    mm->add_instruction(migraphx::make_op("dot"), a, b);

    auto prog = optimize_onnx("matmulnbits_bmm_test.onnx");

    p.sort();
    prog.sort();
    EXPECT(p == prog);
}

// Emits the weight decode chain shared by the dynamic and symbolic broadcast tests below, which
// reuse matmulnbits_bmm_test.onnx with a non-fixed row count. The weights stay static in both, so
// only the final dot differs.
static migraphx::instruction_ref
add_bmm_dequantized_b(migraphx::module& m, migraphx::instruction_ref b, migraphx::instruction_ref s)
{
    b = m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1}}}), b);
    b = m.add_instruction(migraphx::make_op("unpack_int4"), b);
    b = m.add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {8}}}), b);

    s = m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1}}}), s);
    s = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), s);
    s = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 1, 16}}}), s);
    s = m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1}}}), s);
    s = m.add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {8}}}), s);

    auto zp =
        m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::uint8_type, {1}}, {8}});
    zp = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 8}}}), zp);

    b = m.add_instruction(migraphx::make_op("dequantizelinear"), b, s, zp);
    return m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), b);
}

TEST_CASE(matmulnbits_dyn_broadcast_test)
{
    EXPECT(check_parse(
        "matmulnbits_bmm_test.onnx",
        {{"a", {migraphx::shape::float_type, {{2, 2}, {1, 3}, {8, 8}}}},
         {"b", {migraphx::shape::uint8_type, {2, 1, 8}}},
         {"scales", {migraphx::shape::float_type, {2}}}},
        [](migraphx::module& m, const auto& args) {
            auto b  = add_bmm_dequantized_b(m, args[1], args[2]);
            auto b0 = m.add_instruction(migraphx::make_op("broadcast_for_dot"), args[0], b);
            auto b1 = m.add_instruction(migraphx::make_op("broadcast_for_dot"), b, args[0]);
            m.add_return({m.add_instruction(migraphx::make_op("dot"), b0, b1)});
        }));
}

TEST_CASE(matmulnbits_sym_broadcast_test)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;
    EXPECT(check_parse(
        "matmulnbits_bmm_test.onnx",
        {{"a", {migraphx::shape::float_type, sym_dims({lit(2), var("m", {1, 3}), lit(8)})}},
         {"b", {migraphx::shape::uint8_type, {2, 1, 8}}},
         {"scales", {migraphx::shape::float_type, {2}}}},
        [](migraphx::module& m, const auto& args) {
            using migraphx::sym::lit;
            auto b = add_bmm_dequantized_b(m, args[1], args[2]);
            // Only the weights are broadcast: the symbolic row count sits in the dot's contracted
            // dimensions and is carried through untouched.
            b = m.add_instruction(
                migraphx::make_op(
                    "multibroadcast",
                    {{"out_dyn_dims", migraphx::to_value(sym_dims({lit(2), lit(8), lit(2)}))}}),
                b);
            m.add_return({m.add_instruction(migraphx::make_op("dot"), args[0], b)});
        }));
}
