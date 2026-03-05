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

TEST_CASE(addressof_float_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data = {1.0f, 2.5f, -3.0f, 4.5f};
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto lit = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("addressof"), lit);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    // Verify output shape is uint64 with same dimensions
    EXPECT(result.get_shape().type() == migraphx::shape::uint64_type);
    EXPECT(result.get_shape().lens() == s.lens());

    // Verify each address is unique and non-zero
    std::vector<std::size_t> addresses(4);
    result.visit([&](auto output) { addresses.assign(output.begin(), output.end()); });

    for(std::size_t i = 0; i < addresses.size(); ++i)
    {
        EXPECT(addresses[i] != 0);
        for(std::size_t j = i + 1; j < addresses.size(); ++j)
        {
            EXPECT(addresses[i] != addresses[j]);
        }
    }
}

TEST_CASE(addressof_half_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<migraphx::half> data = {
        migraphx::half{1.0f}, migraphx::half{2.5f}, migraphx::half{-3.0f}, migraphx::half{4.5f}};
    migraphx::shape s{migraphx::shape::half_type, {4}};
    auto lit = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("addressof"), lit);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    EXPECT(result.get_shape().type() == migraphx::shape::uint64_type);
    EXPECT(result.get_shape().lens() == s.lens());

    std::vector<std::size_t> addresses(4);
    result.visit([&](auto output) { addresses.assign(output.begin(), output.end()); });

    // For half type, addresses should be 2 bytes apart
    for(std::size_t i = 0; i < addresses.size(); ++i)
    {
        EXPECT(addresses[i] != 0);
    }
}

TEST_CASE(addressof_int32_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<int32_t> data = {-10, 0, 42, 1000, -9999, 123};
    migraphx::shape s{migraphx::shape::int32_type, {2, 3}};
    auto lit = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("addressof"), lit);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    EXPECT(result.get_shape().type() == migraphx::shape::uint64_type);
    EXPECT(result.get_shape().lens() == s.lens());

    std::vector<std::size_t> addresses(6);
    result.visit([&](auto output) { addresses.assign(output.begin(), output.end()); });

    for(const auto& addr : addresses)
    {
        EXPECT(addr != 0);
    }
}

TEST_CASE(addressof_deref_roundtrip_float_test)
{
    // Test that addressof followed by deref returns the original values
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data = {1.0f, 2.5f, -3.0f, 4.5f, 0.0f, -1.25f};
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    auto lit   = mm->add_literal(migraphx::literal{s, data});
    auto addrs = mm->add_instruction(migraphx::make_op("addressof"), lit);
    mm->add_instruction(migraphx::make_op("deref", {{"target_type", migraphx::shape::float_type}}),
                        addrs);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    EXPECT(result.get_shape().type() == migraphx::shape::float_type);
    EXPECT(result.get_shape().lens() == s.lens());

    std::vector<float> results_vector(6);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(results_vector == data);
}

TEST_CASE(addressof_deref_roundtrip_half_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<migraphx::half> data = {
        migraphx::half{1.0f}, migraphx::half{2.5f}, migraphx::half{-3.0f}, migraphx::half{4.5f}};
    migraphx::shape s{migraphx::shape::half_type, {2, 2}};
    auto lit   = mm->add_literal(migraphx::literal{s, data});
    auto addrs = mm->add_instruction(migraphx::make_op("addressof"), lit);
    mm->add_instruction(migraphx::make_op("deref", {{"target_type", migraphx::shape::half_type}}),
                        addrs);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    EXPECT(result.get_shape().type() == migraphx::shape::half_type);

    std::vector<migraphx::half> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(results_vector == data);
}

TEST_CASE(addressof_deref_roundtrip_double_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<double> data = {3.14159265358979, -2.71828182845904, 0.0, 1e-10};
    migraphx::shape s{migraphx::shape::double_type, {4}};
    auto lit   = mm->add_literal(migraphx::literal{s, data});
    auto addrs = mm->add_instruction(migraphx::make_op("addressof"), lit);
    mm->add_instruction(migraphx::make_op("deref", {{"target_type", migraphx::shape::double_type}}),
                        addrs);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    EXPECT(result.get_shape().type() == migraphx::shape::double_type);

    std::vector<double> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(results_vector == data);
}

TEST_CASE(addressof_deref_roundtrip_int32_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<int32_t> data = {-10, 0, 42, 1000, -9999};
    migraphx::shape s{migraphx::shape::int32_type, {5}};
    auto lit   = mm->add_literal(migraphx::literal{s, data});
    auto addrs = mm->add_instruction(migraphx::make_op("addressof"), lit);
    mm->add_instruction(migraphx::make_op("deref", {{"target_type", migraphx::shape::int32_type}}),
                        addrs);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    EXPECT(result.get_shape().type() == migraphx::shape::int32_type);

    std::vector<int32_t> results_vector(5);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(results_vector == data);
}

TEST_CASE(addressof_with_parameter_test)
{
    // Test with parameter instead of literal
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto param = mm->add_parameter("x", s);
    mm->add_instruction(migraphx::make_op("addressof"), param);

    p.compile(migraphx::make_target("ref"));

    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    migraphx::argument input{s, data.data()};
    auto result = p.eval({{"x", input}}).back();

    EXPECT(result.get_shape().type() == migraphx::shape::uint64_type);
    EXPECT(result.get_shape().lens() == s.lens());

    std::vector<std::size_t> addresses(3);
    result.visit([&](auto output) { addresses.assign(output.begin(), output.end()); });

    // Verify addresses point to actual input data
    for(std::size_t i = 0; i < 3; ++i)
    {
        EXPECT(addresses[i] == reinterpret_cast<std::size_t>(&data[i]));
    }
}

TEST_CASE(addressof_deref_with_parameter_test)
{
    // Test roundtrip with parameter
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s{migraphx::shape::float_type, {4}};
    auto param = mm->add_parameter("x", s);
    auto addrs = mm->add_instruction(migraphx::make_op("addressof"), param);
    mm->add_instruction(migraphx::make_op("deref", {{"target_type", migraphx::shape::float_type}}),
                        addrs);

    p.compile(migraphx::make_target("ref"));

    std::vector<float> data = {1.5f, 2.5f, 3.5f, 4.5f};
    migraphx::argument input{s, data.data()};
    auto result = p.eval({{"x", input}}).back();

    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(results_vector == data);
}
