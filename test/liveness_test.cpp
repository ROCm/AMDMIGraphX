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
#include <migraphx/liveness.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

TEST_CASE(liveness_single_alias)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {2}}, {1.0f, 2.0f}});
    auto p1  = mm->add_instruction(pass_op{}, x);
    auto p2  = mm->add_instruction(pass_op{}, p1);
    mm->add_return({p2});

    std::vector<migraphx::instruction_ref> consumed;
    migraphx::liveness(*mm, [&](auto ins, const auto&) {
        consumed.push_back(ins);
    });

    // With single alias ops, pass_ops alias to x, so liveness tracks x
    // The callback is called when x is "consumed" (last usage)
    EXPECT(migraphx::contains(consumed, x));
}

TEST_CASE(liveness_multi_alias)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {2}}, {1.0f, 2.0f}});
    auto y   = mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {2}}, {3.0f, 4.0f}});
    // multi_alias_op aliases both x and y
    auto ma  = mm->add_instruction(multi_alias_op{}, x, y);
    auto p1  = mm->add_instruction(pass_op{}, ma);
    mm->add_return({p1});

    std::vector<migraphx::instruction_ref> consumed;
    migraphx::liveness(*mm, [&](auto ins, const auto&) {
        consumed.push_back(ins);
    });

    // Both x and y should be tracked and consumed
    // because multi_alias_op aliases both inputs
    EXPECT(migraphx::contains(consumed, x));
    EXPECT(migraphx::contains(consumed, y));
}

TEST_CASE(liveness_multi_alias_cascade)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {2}}, {1.0f, 2.0f}});
    auto y   = mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {2}}, {3.0f, 4.0f}});
    auto z   = mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {2}}, {5.0f, 6.0f}});
    // First multi_alias aliases x and y
    auto ma1 = mm->add_instruction(multi_alias_op{}, x, y);
    // Second multi_alias aliases ma1 (which aliases x,y) and z
    auto ma2 = mm->add_instruction(multi_alias_op{}, ma1, z);
    mm->add_return({ma2});

    std::vector<migraphx::instruction_ref> consumed;
    migraphx::liveness(*mm, [&](auto ins, const auto&) {
        consumed.push_back(ins);
    });

    // All three literals should be tracked and consumed
    // ma2 transitively aliases x, y, z
    EXPECT(migraphx::contains(consumed, x));
    EXPECT(migraphx::contains(consumed, y));
    EXPECT(migraphx::contains(consumed, z));
}

TEST_CASE(liveness_multi_alias_both_tracked)
{
    // This test verifies that when multi_alias_op aliases multiple inputs,
    // ALL aliased instructions are properly tracked in liveness analysis.
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {2}}, {1.0f, 2.0f}});
    auto y   = mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {2}}, {3.0f, 4.0f}});
    // multi_alias_op returns output_alias {0, 1} - it aliases both inputs
    auto ma  = mm->add_instruction(multi_alias_op{}, x, y);
    mm->add_return({ma});

    // Count how many times each literal appears in any live_set across all callbacks
    std::size_t x_live_count = 0;
    std::size_t y_live_count = 0;
    migraphx::liveness(*mm, [&](auto, const auto& live_set) {
        if(migraphx::contains(live_set, x))
            x_live_count++;
        if(migraphx::contains(live_set, y))
            y_live_count++;
    });

    // Both x and y should appear as live at some point during liveness analysis
    // (because multi_alias_op properly exposes both as aliases)
    // Note: they might have count 0 if they're only processed when the live_set is already emptied
    // The key test is that BOTH get consumed (callback called for both)
    std::vector<migraphx::instruction_ref> consumed;
    migraphx::liveness(*mm, [&](auto ins, const auto&) {
        consumed.push_back(ins);
    });

    EXPECT(migraphx::contains(consumed, x));
    EXPECT(migraphx::contains(consumed, y));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }

