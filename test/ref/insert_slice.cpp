/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(insert_slice_single_offset_axis_0)
{
    // dest shape [6], src shape [2], offset 2 -> copy src into dest at indices 2,3
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto dtype = migraphx::shape::float_type;
    auto otype = migraphx::shape::int64_type;

    std::vector<float> src_data{10.0f, 20.0f};
    std::vector<float> dest_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<int64_t> off_data{2};

    migraphx::shape src_shape{dtype, {2}};
    migraphx::shape dest_shape{dtype, {6}};
    migraphx::shape off_shape{otype, {1}};

    auto src  = mm->add_literal(migraphx::literal{src_shape, src_data});
    auto off  = mm->add_literal(migraphx::literal{off_shape, off_data});
    auto dest = mm->add_literal(migraphx::literal{dest_shape, dest_data});
    auto r    = mm->add_instruction(
        migraphx::make_op("insert_slice", {{"axis", 0}}), src, off, dest);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> result_vec;
    result.visit([&](auto out) { result_vec.assign(out.begin(), out.end()); });

    std::vector<float> gold{1.0f, 2.0f, 10.0f, 20.0f, 5.0f, 6.0f};
    EXPECT(migraphx::verify::verify_rms_range(result_vec, gold));
}

TEST_CASE(insert_slice_single_offset_axis_1)
{
    // dest [2, 8, 3], src [2, 2, 3], axis=1, offset=3 -> dest[:, 3:5, :] = src
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto dtype = migraphx::shape::float_type;
    auto otype = migraphx::shape::int64_type;

    migraphx::shape src_shape{dtype, {2, 2, 3}};
    migraphx::shape dest_shape{dtype, {2, 8, 3}};
    migraphx::shape off_shape{otype, {1}};

    std::vector<float> src_data(2 * 2 * 3, 1.0f);
    std::vector<float> dest_data(2 * 8 * 3, 0.0f);
    std::vector<int64_t> off_data{3};

    auto src  = mm->add_literal(migraphx::literal{src_shape, src_data});
    auto off  = mm->add_literal(migraphx::literal{off_shape, off_data});
    auto dest = mm->add_literal(migraphx::literal{dest_shape, dest_data});
    auto r    = mm->add_instruction(
        migraphx::make_op("insert_slice", {{"axis", 1}}), src, off, dest);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> result_vec;
    result.visit([&](auto out) { result_vec.assign(out.begin(), out.end()); });

    // Check: positions [0:3] and [5:8] along axis 1 stay 0; [3:5] are 1
    for(std::size_t b = 0; b < 2; b++)
        for(std::size_t i = 0; i < 8; i++)
            for(std::size_t k = 0; k < 3; k++)
            {
                std::size_t idx = b * 8 * 3 + i * 3 + k;
                float expected   = (i >= 3 and i < 5) ? 1.0f : 0.0f;
                EXPECT(result_vec[idx] == expected);
            }
}

TEST_CASE(insert_slice_multi_offset_axis_2)
{
    // dest [2, 2, 6], src [2, 2, 2], axis=2. Offsets [0, 2] per batch row -> row0 at 0:2, row1 at 2:4
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto dtype = migraphx::shape::float_type;
    auto otype = migraphx::shape::int64_type;

    migraphx::shape src_shape{dtype, {2, 2, 2}};
    migraphx::shape dest_shape{dtype, {2, 2, 6}};
    migraphx::shape off_shape{otype, {2, 2}};  // num_outer = 2*2 = 4

    std::vector<float> src_data{1, 2, 3, 4, 5, 6, 7, 8};  // 2*2*2
    std::vector<float> dest_data(2 * 2 * 6, 0.0f);
    std::vector<int64_t> off_data{0, 2, 1, 3};  // 4 offsets for 4 "rows"

    auto src  = mm->add_literal(migraphx::literal{src_shape, src_data});
    auto off  = mm->add_literal(migraphx::literal{off_shape, off_data});
    auto dest = mm->add_literal(migraphx::literal{dest_shape, dest_data});
    auto r    = mm->add_instruction(
        migraphx::make_op("insert_slice", {{"axis", 2}}), src, off, dest);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> result_vec;
    result.visit([&](auto out) { result_vec.assign(out.begin(), out.end()); });

    // dest layout [2,2,6]: (b,i,j) -> b*12 + i*6 + j
    // (0,0,:): offset 0 -> dest[0,0,0:2] = src[0,0,:] = 1,2
    EXPECT(result_vec[0] == 1.0f);
    EXPECT(result_vec[1] == 2.0f);
    // (0,1,:): offset 2 -> dest[0,1,2:4] = src[0,1,:] = 3,4  -> linear 1*6+2, 1*6+3
    EXPECT(result_vec[1 * 6 + 2] == 3.0f);
    EXPECT(result_vec[1 * 6 + 3] == 4.0f);
    // (1,0,:): offset 1 -> dest[1,0,1:3] = src[1,0,:] = 5,6
    EXPECT(result_vec[12 + 1] == 5.0f);
    EXPECT(result_vec[12 + 2] == 6.0f);
    // (1,1,:): offset 3 -> dest[1,1,3:5] = src[1,1,:] = 7,8
    EXPECT(result_vec[12 + 6 + 3] == 7.0f);
    EXPECT(result_vec[12 + 6 + 4] == 8.0f);
}

TEST_CASE(insert_slice_1d_single_offset)
{
    // Simple 1D: dest length 5, src length 2, offset 1
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto dtype = migraphx::shape::float_type;
    auto otype = migraphx::shape::int64_type;

    migraphx::shape src_shape{dtype, {2}};
    migraphx::shape dest_shape{dtype, {5}};
    migraphx::shape off_shape{otype, {1}};

    std::vector<float> src_data{-1.0f, -2.0f};
    std::vector<float> dest_data{10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    std::vector<int64_t> off_data{1};

    auto src  = mm->add_literal(migraphx::literal{src_shape, src_data});
    auto off  = mm->add_literal(migraphx::literal{off_shape, off_data});
    auto dest = mm->add_literal(migraphx::literal{dest_shape, dest_data});
    auto r    = mm->add_instruction(
        migraphx::make_op("insert_slice", {{"axis", 0}}), src, off, dest);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> result_vec;
    result.visit([&](auto out) { result_vec.assign(out.begin(), out.end()); });

    std::vector<float> gold{10.0f, -1.0f, -2.0f, 40.0f, 50.0f};
    EXPECT(migraphx::verify::verify_rms_range(result_vec, gold));
}

TEST_CASE(insert_slice_deref_cpu_throws)
{
    // When deref=true, CPU compute should throw
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto dtype = migraphx::shape::float_type;
    auto otype = migraphx::shape::int64_type;
    auto utype = migraphx::shape::uint64_type;

    migraphx::shape src_shape{dtype, {2}};
    migraphx::shape dest_shape{utype, {2}};  // uint for deref
    migraphx::shape off_shape{otype, {1}};

    std::vector<float> src_data{1.0f, 2.0f};
    std::vector<uint64_t> dest_data{0, 0};
    std::vector<int64_t> off_data{0};

    auto src  = mm->add_literal(migraphx::literal{src_shape, src_data});
    auto off  = mm->add_literal(migraphx::literal{off_shape, off_data});
    auto dest = mm->add_literal(migraphx::literal{dest_shape, dest_data});
    auto r    = mm->add_instruction(
        migraphx::make_op("insert_slice", {{"axis", 0}, {"deref", true}}), src, off, dest);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));
    EXPECT(test::throws([&] { p.eval({}); }));
}
