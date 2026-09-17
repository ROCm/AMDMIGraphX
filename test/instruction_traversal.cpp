/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/instruction_traversal.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

using instruction_refs = std::vector<migraphx::instruction_ref>;

template <class Range>
static instruction_refs collect(const Range& r)
{
    return {r.begin(), r.end()};
}

static migraphx::instruction_ref add_allocate(migraphx::module& m, const migraphx::shape& s)
{
    return m.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
}

TEST_CASE(output_path_linear)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m;
    auto x  = m.add_parameter("x", s);
    auto p1 = m.add_instruction(pass_op{}, x);
    auto p2 = m.add_instruction(pass_op{}, p1);
    auto p3 = m.add_instruction(pass_op{}, p2);

    EXPECT(collect(migraphx::get_output_path(x)) == instruction_refs{x, p1, p2, p3});
    EXPECT(collect(migraphx::get_output_path(p2)) == instruction_refs{p2, p3});
}

// The path cannot be followed past an instruction that is used more than once
TEST_CASE(output_path_multiple_outputs)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m;
    auto x  = m.add_parameter("x", s);
    auto p1 = m.add_instruction(pass_op{}, x);
    m.add_instruction(pass_op{}, p1);
    m.add_instruction(pass_op{}, p1);

    EXPECT(collect(migraphx::get_output_path(x)) == instruction_refs{x, p1});
}

TEST_CASE(output_path_no_outputs)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m;
    auto x = m.add_parameter("x", s);

    EXPECT(collect(migraphx::get_output_path(x)) == instruction_refs{x});
}

TEST_CASE(alias_path_allocation)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m;
    auto alloc = add_allocate(m, s);
    auto p1    = m.add_instruction(pass_op{}, alloc);
    auto p2    = m.add_instruction(pass_op{}, p1);

    EXPECT(collect(migraphx::get_alias_path(p2)) == instruction_refs{p2, p1, alloc});
    EXPECT(collect(migraphx::get_alias_path(alloc)) == instruction_refs{alloc});
}

// The buffer can be owned by a parameter instead of an allocation
TEST_CASE(alias_path_parameter)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m;
    auto x  = m.add_parameter("x", s);
    auto p1 = m.add_instruction(pass_op{}, x);

    EXPECT(collect(migraphx::get_alias_path(p1)) == instruction_refs{p1, x});
}

TEST_CASE(alias_path_shape_transforms)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 4, 1}};
    migraphx::module m;
    auto alloc = add_allocate(m, s);
    auto p1    = m.add_instruction(pass_op{}, alloc);
    auto sq    = m.add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), p1);
    auto t     = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), sq);

    EXPECT(collect(migraphx::get_alias_path(t)) == instruction_refs{t, sq, p1, alloc});
}

// An operator that does not alias its input owns its own buffer
TEST_CASE(alias_path_no_alias)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m;
    auto x   = m.add_parameter("x", s);
    auto y   = m.add_parameter("y", s);
    auto sum = m.add_instruction(sum_op{}, x, y);

    EXPECT(collect(migraphx::get_alias_path(sum)) == instruction_refs{sum});
}

// There is no single buffer to follow when more than one input is aliased
TEST_CASE(alias_path_multiple_aliases)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m;
    auto x  = m.add_parameter("x", s);
    auto y  = m.add_parameter("y", s);
    auto ma = m.add_instruction(multi_alias_op{}, x, y);

    EXPECT(collect(migraphx::get_alias_path(ma)) == instruction_refs{ma});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
