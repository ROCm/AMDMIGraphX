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
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(insert_slice_regular_float)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> source_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> dest_data(8, 0.0f);

    migraphx::shape s{migraphx::shape::float_type, {2, 4}};
    auto source_lit = mm->add_literal(migraphx::literal{s, source_data});
    auto dest_lit   = mm->add_literal(migraphx::literal{s, dest_data});

    mm->add_instruction(
        migraphx::make_op("insert_slice",
                         {{"static_offsets", {0, 0}}, {"static_strides", {1, 1}}, {"deref_dest", false}}),
        source_lit,
        dest_lit);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector(8);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, source_data));
}

TEST_CASE(insert_slice_regular_half)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<migraphx::half> source_data = {
        migraphx::half{1.0f}, migraphx::half{2.0f}, migraphx::half{3.0f}, migraphx::half{4.0f},
        migraphx::half{5.0f}, migraphx::half{6.0f}, migraphx::half{7.0f}, migraphx::half{8.0f}};
    std::vector<migraphx::half> dest_data(8, migraphx::half{0.0f});

    migraphx::shape s{migraphx::shape::half_type, {2, 4}};
    auto source_lit = mm->add_literal(migraphx::literal{s, source_data});
    auto dest_lit   = mm->add_literal(migraphx::literal{s, dest_data});

    mm->add_instruction(
        migraphx::make_op("insert_slice",
                         {{"static_offsets", {0, 0}}, {"static_strides", {1, 1}}, {"deref_dest", false}}),
        source_lit,
        dest_lit);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<migraphx::half> results_vector(8);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, source_data));
}

TEST_CASE(insert_slice_deref_dest_float)
{
    // Destination holds pointers to a writable buffer. insert_slice(deref_dest=true)
    // writes source through those pointers. Verify with deref operator.
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> buffer(4, 0.0f);
    std::vector<std::size_t> ptr_data(4);
    for(std::size_t i = 0; i < 4; ++i)
        ptr_data[i] = reinterpret_cast<std::size_t>(&buffer[i]);

    std::vector<float> source_data = {10.0f, 20.0f, 30.0f, 40.0f};

    migraphx::shape ptr_shape{migraphx::shape::uint64_type, {4}};
    migraphx::shape src_shape{migraphx::shape::float_type, {4}};

    auto dest_lit   = mm->add_literal(migraphx::literal{ptr_shape, ptr_data});
    auto source_lit = mm->add_literal(migraphx::literal{src_shape, source_data});

    auto out = mm->add_instruction(
        migraphx::make_op("insert_slice",
                         {{"static_offsets", {0}}, {"static_strides", {1}}, {"deref_dest", true}}),
        source_lit,
        dest_lit);
    mm->add_instruction(
        migraphx::make_op("deref", {{"target_type", migraphx::shape::float_type}}), out);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, source_data));
}

TEST_CASE(insert_slice_deref_dest_half)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<migraphx::half> buffer(4, migraphx::half{0.0f});
    std::vector<std::size_t> ptr_data(4);
    for(std::size_t i = 0; i < 4; ++i)
        ptr_data[i] = reinterpret_cast<std::size_t>(&buffer[i]);

    std::vector<migraphx::half> source_data = {
        migraphx::half{1.0f}, migraphx::half{-2.0f}, migraphx::half{3.0f}, migraphx::half{4.0f}};

    migraphx::shape ptr_shape{migraphx::shape::uint64_type, {4}};
    migraphx::shape src_shape{migraphx::shape::half_type, {4}};

    auto dest_lit   = mm->add_literal(migraphx::literal{ptr_shape, ptr_data});
    auto source_lit = mm->add_literal(migraphx::literal{src_shape, source_data});

    auto out = mm->add_instruction(
        migraphx::make_op("insert_slice",
                         {{"static_offsets", {0}}, {"static_strides", {1}}, {"deref_dest", true}}),
        source_lit,
        dest_lit);
    mm->add_instruction(
        migraphx::make_op("deref", {{"target_type", migraphx::shape::half_type}}), out);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<migraphx::half> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, source_data));
}

TEST_CASE(insert_slice_batched_offsets_input)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> source_data(4, 1.0f);
    std::vector<float> dest_data(6, 0.0f);
    std::vector<int64_t> offsets_data = {0, 0, 0, 1};

    migraphx::shape src_shape{migraphx::shape::float_type, {2, 2}};
    migraphx::shape dest_shape{migraphx::shape::float_type, {2, 3}};
    migraphx::shape offsets_shape{migraphx::shape::int64_type, {2, 2}};

    auto source_lit  = mm->add_literal(migraphx::literal{src_shape, source_data});
    auto dest_lit    = mm->add_literal(migraphx::literal{dest_shape, dest_data});
    auto offsets_lit = mm->add_literal(migraphx::literal{offsets_shape, offsets_data});

    mm->add_instruction(
        migraphx::make_op("insert_slice",
                         {{"static_strides", {1, 1}}, {"deref_dest", false}}),
        source_lit,
        dest_lit,
        offsets_lit);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> expected = {1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f};
    std::vector<float> results_vector(6);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, expected));
}

TEST_CASE(insert_slice_with_strides)
{
    // Scatter source into output at dest_idx = src_idx * strides + offsets.
    // offsets={0,1}, strides={1,1}: write source[i,j] to output[i, j+1].
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> source_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> dest_data(6, 0.0f); // 2x3 dest

    migraphx::shape src_shape{migraphx::shape::float_type, {2, 2}};
    migraphx::shape dest_shape{migraphx::shape::float_type, {2, 3}};

    auto source_lit = mm->add_literal(migraphx::literal{src_shape, source_data});
    auto dest_lit   = mm->add_literal(migraphx::literal{dest_shape, dest_data});

    mm->add_instruction(
        migraphx::make_op("insert_slice",
                         {{"static_offsets", {0, 1}}, {"static_strides", {1, 1}}, {"deref_dest", false}}),
        source_lit,
        dest_lit);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    // source[0,0]->out[0,1], source[0,1]->out[0,2], source[1,0]->out[1,1], source[1,1]->out[1,2]
    std::vector<float> expected = {0.0f, 1.0f, 2.0f, 0.0f, 3.0f, 4.0f};
    std::vector<float> results_vector(6);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, expected));
}
