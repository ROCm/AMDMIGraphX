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
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <test.hpp>
#include <basic_ops.hpp>

TEST_CASE(simple_alias)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l   = mm->add_literal(1);
    auto p1  = mm->add_instruction(pass_op{}, l);
    EXPECT(migraphx::contains(migraphx::instruction::get_output_alias(l), l));
    EXPECT(migraphx::contains(migraphx::instruction::get_output_alias(p1), l));
}

TEST_CASE(cascade_alias)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l   = mm->add_literal(1);
    auto p1  = mm->add_instruction(pass_op{}, l);
    auto p2  = mm->add_instruction(pass_op{}, p1);
    auto p3  = mm->add_instruction(pass_op{}, p2);
    EXPECT(migraphx::contains(migraphx::instruction::get_output_alias(l), l));
    EXPECT(migraphx::contains(migraphx::instruction::get_output_alias(p1), l));
    EXPECT(migraphx::contains(migraphx::instruction::get_output_alias(p2), l));
    EXPECT(migraphx::contains(migraphx::instruction::get_output_alias(p3), l));
}

TEST_CASE(no_alias)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(1);
    auto y   = mm->add_literal(2);
    auto sum = mm->add_instruction(sum_op{}, x, y);
    EXPECT(migraphx::contains(migraphx::instruction::get_output_alias(sum), sum));
}

TEST_CASE(multiple_aliases)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto x       = mm->add_literal(1);
    auto y       = mm->add_literal(2);
    auto ma      = mm->add_instruction(multi_alias_op{}, x, y);
    auto aliases = migraphx::instruction::get_output_alias(ma);
    // multi_alias_op aliases both inputs, so we should get both literals back
    EXPECT(aliases.size() == 2);
    EXPECT(migraphx::contains(aliases, x));
    EXPECT(migraphx::contains(aliases, y));
}

TEST_CASE(multiple_aliases_shallow)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(1);
    auto y   = mm->add_literal(2);
    auto p1  = mm->add_instruction(pass_op{}, x);
    auto p2  = mm->add_instruction(pass_op{}, y);
    auto ma  = mm->add_instruction(multi_alias_op{}, p1, p2);
    // shallow=true returns immediate inputs (p1, p2), not root aliases
    auto shallow_aliases = migraphx::instruction::get_output_alias(ma, true);
    EXPECT(shallow_aliases.size() == 2);
    EXPECT(migraphx::contains(shallow_aliases, p1));
    EXPECT(migraphx::contains(shallow_aliases, p2));
    // shallow=false (default) returns root aliases (x, y)
    auto deep_aliases = migraphx::instruction::get_output_alias(ma);
    EXPECT(deep_aliases.size() == 2);
    EXPECT(migraphx::contains(deep_aliases, x));
    EXPECT(migraphx::contains(deep_aliases, y));
}

TEST_CASE(multiple_aliases_cascade)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(1);
    auto y   = mm->add_literal(2);
    auto z   = mm->add_literal(3);
    // First multi_alias aliases x and y
    auto ma1 = mm->add_instruction(multi_alias_op{}, x, y);
    // Second multi_alias aliases ma1 and z
    auto ma2 = mm->add_instruction(multi_alias_op{}, ma1, z);
    // Should recursively expand to get x, y, z
    auto aliases = migraphx::instruction::get_output_alias(ma2);
    EXPECT(aliases.size() == 3);
    EXPECT(migraphx::contains(aliases, x));
    EXPECT(migraphx::contains(aliases, y));
    EXPECT(migraphx::contains(aliases, z));
}

TEST_CASE(alias_vector_size)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l   = mm->add_literal(1);
    // No alias - returns vector with self
    auto aliases_self = migraphx::instruction::get_output_alias(l);
    EXPECT(aliases_self.size() == 1);
    // Single alias - returns vector with one element
    auto p1             = mm->add_instruction(pass_op{}, l);
    auto aliases_single = migraphx::instruction::get_output_alias(p1);
    EXPECT(aliases_single.size() == 1);
    // Multiple aliases - returns vector with multiple elements
    auto x             = mm->add_literal(2);
    auto ma            = mm->add_instruction(multi_alias_op{}, l, x);
    auto aliases_multi = migraphx::instruction::get_output_alias(ma);
    EXPECT(aliases_multi.size() == 2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
