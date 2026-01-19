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

TEST_CASE(deref_half_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    // Create source data to be dereferenced
    std::vector<migraphx::half> source_data = {
        migraphx::half{1.0f}, migraphx::half{2.5f}, migraphx::half{-3.0f}, migraphx::half{4.5f}};

    // Create pointer values (addresses of the source data elements)
    std::vector<std::size_t> ptr_data(4);
    for(std::size_t i = 0; i < 4; ++i)
    {
        ptr_data[i] = reinterpret_cast<std::size_t>(&source_data[i]);
    }

    migraphx::shape ptr_shape{migraphx::shape::uint64_type, {2, 2}};
    auto ptr_lit = mm->add_literal(migraphx::literal{ptr_shape, ptr_data});
    mm->add_instruction(migraphx::make_op("deref", {{"target_type", migraphx::shape::half_type}}),
                        ptr_lit);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<migraphx::half> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(results_vector == source_data);
}

TEST_CASE(deref_float_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    // Create source data to be dereferenced
    std::vector<float> source_data = {1.0f, 2.5f, -3.0f, 4.5f, 0.0f, -1.25f};

    // Create pointer values (addresses of the source data elements)
    std::vector<std::size_t> ptr_data(6);
    for(std::size_t i = 0; i < 6; ++i)
    {
        ptr_data[i] = reinterpret_cast<std::size_t>(&source_data[i]);
    }

    migraphx::shape ptr_shape{migraphx::shape::uint64_type, {2, 3}};
    auto ptr_lit = mm->add_literal(migraphx::literal{ptr_shape, ptr_data});
    mm->add_instruction(migraphx::make_op("deref", {{"target_type", migraphx::shape::float_type}}),
                        ptr_lit);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector(6);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(results_vector == source_data);
}

TEST_CASE(deref_int32_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    // Create source data to be dereferenced
    std::vector<int32_t> source_data = {-10, 0, 42, 1000, -9999};

    // Create pointer values (addresses of the source data elements)
    std::vector<std::size_t> ptr_data(5);
    for(std::size_t i = 0; i < 5; ++i)
    {
        ptr_data[i] = reinterpret_cast<std::size_t>(&source_data[i]);
    }

    migraphx::shape ptr_shape{migraphx::shape::uint64_type, {5}};
    auto ptr_lit = mm->add_literal(migraphx::literal{ptr_shape, ptr_data});
    mm->add_instruction(migraphx::make_op("deref", {{"target_type", migraphx::shape::int32_type}}),
                        ptr_lit);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<int32_t> results_vector(5);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(results_vector == source_data);
}

TEST_CASE(deref_double_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    // Create source data to be dereferenced
    std::vector<double> source_data = {3.14159265358979, -2.71828182845904, 0.0, 1e-10};

    // Create pointer values (addresses of the source data elements)
    std::vector<std::size_t> ptr_data(4);
    for(std::size_t i = 0; i < 4; ++i)
    {
        ptr_data[i] = reinterpret_cast<std::size_t>(&source_data[i]);
    }

    migraphx::shape ptr_shape{migraphx::shape::uint64_type, {4}};
    auto ptr_lit = mm->add_literal(migraphx::literal{ptr_shape, ptr_data});
    mm->add_instruction(migraphx::make_op("deref", {{"target_type", migraphx::shape::double_type}}),
                        ptr_lit);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<double> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(results_vector == source_data);
}

TEST_CASE(deref_noncontiguous_pointers_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    // Create source data with more elements than we'll access
    std::vector<float> source_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    // Create pointer values pointing to non-contiguous elements (every other element)
    std::vector<std::size_t> ptr_data(4);
    ptr_data[0] = reinterpret_cast<std::size_t>(&source_data[0]);
    ptr_data[1] = reinterpret_cast<std::size_t>(&source_data[2]);
    ptr_data[2] = reinterpret_cast<std::size_t>(&source_data[4]);
    ptr_data[3] = reinterpret_cast<std::size_t>(&source_data[6]);

    migraphx::shape ptr_shape{migraphx::shape::uint64_type, {2, 2}};
    auto ptr_lit = mm->add_literal(migraphx::literal{ptr_shape, ptr_data});
    mm->add_instruction(migraphx::make_op("deref", {{"target_type", migraphx::shape::float_type}}),
                        ptr_lit);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    std::vector<float> expected = {1.0f, 3.0f, 5.0f, 7.0f};
    EXPECT(results_vector == expected);
}

TEST_CASE(deref_uint8_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    // Create source data to be dereferenced
    std::vector<uint8_t> source_data = {0, 128, 255, 1, 127};

    // Create pointer values (addresses of the source data elements)
    std::vector<std::size_t> ptr_data(5);
    for(std::size_t i = 0; i < 5; ++i)
    {
        ptr_data[i] = reinterpret_cast<std::size_t>(&source_data[i]);
    }

    migraphx::shape ptr_shape{migraphx::shape::uint64_type, {5}};
    auto ptr_lit = mm->add_literal(migraphx::literal{ptr_shape, ptr_data});
    mm->add_instruction(migraphx::make_op("deref", {{"target_type", migraphx::shape::uint8_type}}),
                        ptr_lit);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<uint8_t> results_vector(5);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(results_vector == source_data);
}

TEST_CASE(deref_int64_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    // Create source data to be dereferenced
    // Note: Values must be exactly representable as double (up to 2^53)
    // since the unary base class converts through double
    std::vector<int64_t> source_data = {-1000000000000LL, -1, 0, 1, 1000000000000LL};

    // Create pointer values (addresses of the source data elements)
    std::vector<std::size_t> ptr_data(5);
    for(std::size_t i = 0; i < 5; ++i)
    {
        ptr_data[i] = reinterpret_cast<std::size_t>(&source_data[i]);
    }

    migraphx::shape ptr_shape{migraphx::shape::uint64_type, {5}};
    auto ptr_lit = mm->add_literal(migraphx::literal{ptr_shape, ptr_data});
    mm->add_instruction(migraphx::make_op("deref", {{"target_type", migraphx::shape::int64_type}}),
                        ptr_lit);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<int64_t> results_vector(5);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(results_vector == source_data);
}
