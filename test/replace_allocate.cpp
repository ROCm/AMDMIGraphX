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
#include <migraphx/allocation_model.hpp>
#include <migraphx/replace_allocate.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/register_op.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct test_copy : migraphx::auto_register_op<test_copy>
{
    std::string name() const { return "test_copy"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs) const
    {
        migraphx::check_shapes{inputs, *this}.has(2);
        return inputs.back();
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape&,
                               const std::vector<migraphx::argument>& inputs) const
    {
        inputs.at(0).visit([&](auto x) {
            inputs.at(1).visit([&](auto y) { std::copy(x.begin(), x.end(), y.begin()); });
        });
        return inputs.back();
    }

    std::vector<std::size_t> output_alias(const std::vector<migraphx::shape>&) const { return {1}; }
};

struct allocate_no_out : migraphx::auto_register_op<allocate_no_out>
{
    migraphx::shape s{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.s, "shape"));
    }

    std::string name() const { return "allocate_no_out"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs) const
    {
        migraphx::check_shapes{inputs, *this}.has(0);
        return s;
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape& output_shape,
                               const std::vector<migraphx::argument>&) const
    {
        return migraphx::argument{output_shape};
    }
};

struct allocate_with_out : migraphx::auto_register_op<allocate_with_out>
{
    migraphx::shape s{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.s, "shape"));
    }

    std::string name() const { return "allocate_with_out"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs) const
    {
        migraphx::check_shapes{inputs, *this}.has(0);
        return s;
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape& output_shape,
                               const std::vector<migraphx::argument>&) const
    {
        return migraphx::argument{output_shape};
    }
};

// allocation model that has no out params
struct allocation_no_out_model
{
    std::string name() const { return "allocate_no_out"; }
    migraphx::operation allocate(const migraphx::shape& s) const
    {
        return migraphx::make_op(name(), {{"shape", to_value(s)}});
    }
    migraphx::operation preallocate(const migraphx::shape&, const std::string&) const { return {}; }
    std::string copy() const { return "test_copy"; }
    bool needs_out_params() const { return false; }
};

// allocation model with out params
struct allocation_with_out_model
{
    std::string name() const { return "allocate_with_out"; }
    migraphx::operation allocate(const migraphx::shape& s) const
    {
        return migraphx::make_op(name(), {{"shape", to_value(s)}});
    }
    migraphx::operation preallocate(const migraphx::shape&, const std::string&) const { return {}; }
    std::string copy() const { return "test_copy"; }
    bool needs_out_params() const { return true; }
};

static void
run_pass(migraphx::module& m, migraphx::allocation_model model, bool offload_copy = false)
{
    migraphx::run_passes(m,
                         {migraphx::replace_allocate{std::move(model), offload_copy},
                          migraphx::dead_code_elimination{}});
}

static void
run_pass(migraphx::program& p, migraphx::allocation_model model, bool offload_copy = false)
{
    migraphx::run_passes(p,
                         {migraphx::replace_allocate{std::move(model), offload_copy},
                          migraphx::dead_code_elimination{}});
}

static migraphx::module create_simple_program()
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {5}};
    auto x = m.add_parameter("x", s);
    auto y = m.add_parameter("y", s);
    auto alloc =
        m.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
    m.add_instruction(pass_op{}, alloc, x, y);
    return m;
}

TEST_CASE(allocate_no_out)
{
    migraphx::module m = create_simple_program();
    run_pass(m, allocation_no_out_model{});

    EXPECT(std::any_of(m.begin(), m.end(), [](const migraphx::instruction& ins) {
        return migraphx::contains(ins.name(), "allocate_no_out");
    }));
}

TEST_CASE(allocate_with_out_param)
{
    migraphx::module m = create_simple_program();
    run_pass(m, allocation_with_out_model{});

    EXPECT(std::none_of(m.begin(), m.end(), [](const migraphx::instruction& ins) {
        return migraphx::contains(ins.name(), "allocate");
    }));
}

TEST_CASE(allocate_with_out_return)
{
    migraphx::module m = create_simple_program();
    m.add_return({std::prev(m.end())});
    run_pass(m, allocation_with_out_model{});

    EXPECT(std::none_of(m.begin(), m.end(), [](const migraphx::instruction& ins) {
        return migraphx::contains(ins.name(), "allocate");
    }));
}

TEST_CASE(allocate_with_out_no_params)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {5}};
    auto x = m.add_parameter("x", s);
    auto y = m.add_parameter("y", s);
    auto z = m.add_parameter("z", s);
    auto alloc =
        m.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
    auto pass1 = m.add_instruction(pass_op{}, alloc, x, y);
    auto alloc2 =
        m.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
    m.add_instruction(pass_op{}, alloc2, z, pass1);
    run_pass(m, allocation_with_out_model{});

    EXPECT(std::any_of(m.begin(), m.end(), [](const migraphx::instruction& ins) {
        return migraphx::contains(ins.name(), "allocate_with_out");
    }));
}

TEST_CASE(if_allocate)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape cond_s{migraphx::shape::bool_type};
    auto cond = mm->add_parameter("cond", cond_s);
    migraphx::shape s{migraphx::shape::float_type, {5}};
    auto x = mm->add_parameter("x", s);
    auto y = mm->add_parameter("y", s);

    auto* then_mod = p.create_module("If_0_if");
    auto alloc     = then_mod->add_instruction(
        migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
    auto a1 = then_mod->add_instruction(pass_op{}, alloc, x);
    then_mod->add_return({a1});

    auto* else_mod = p.create_module("If_0_else");
    auto alloc1    = else_mod->add_instruction(
        migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
    auto a2 = else_mod->add_instruction(pass_op{}, alloc1, y);
    else_mod->add_return({a2});

    mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});

    run_pass(p, allocation_with_out_model{});
    EXPECT(std::any_of(mm->begin(), mm->end(), [](const migraphx::instruction& ins) {
        return migraphx::contains(ins.name(), "allocate_with_out");
    }));
}

TEST_CASE(allocate_copy_with_out)
{
    migraphx::shape s{migraphx::shape::float_type, {5}};
    migraphx::module m1;
    {
        auto x     = m1.add_parameter("x", s);
        auto y     = m1.add_parameter("y", s);
        auto pass  = m1.add_instruction(tuple_op{}, x, y);
        auto elem1 = m1.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), pass);
        m1.add_return({elem1});
    }
    run_pass(m1, allocation_with_out_model{});
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s);
        auto y      = m2.add_parameter("y", s);
        auto output = m2.add_parameter("output", s);
        auto pass   = m2.add_instruction(tuple_op{}, x, y);
        auto elem1  = m2.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), pass);
        auto copy   = m2.add_instruction(migraphx::make_op("test_copy"), elem1, output);
        m2.add_return({copy});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(allocate_copy_with_no_out)
{
    migraphx::shape s{migraphx::shape::float_type, {5}};
    migraphx::module m1;
    {
        auto x     = m1.add_parameter("x", s);
        auto y     = m1.add_parameter("y", s);
        auto pass  = m1.add_instruction(tuple_op{}, x, y);
        auto elem1 = m1.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), pass);
        m1.add_return({elem1});
    }
    migraphx::module m2 = m1;
    run_pass(m1, allocation_no_out_model{});
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(allocate_out_multi_return_partial_alloc)
{
    migraphx::shape s{migraphx::shape::float_type, {5}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s);
        auto alloc =
            m1.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
        auto p1 = m1.add_instruction(pass_op{}, alloc);
        m1.add_return({x, p1});
    }
    run_pass(m1, allocation_with_out_model{});

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s);
        auto output = m2.add_parameter("output_1", s);
        auto p1     = m2.add_instruction(pass_op{}, output);
        m2.add_return({x, p1});
    }
    EXPECT(m1.sort() == m2.sort());
}

// Test that replace_allocate handles multi-alias operations correctly
// when checking for shape matches (insert_copy code path)
TEST_CASE(multi_alias_shape_check)
{
    migraphx::shape s{migraphx::shape::float_type, {5}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s);
        auto y = m1.add_parameter("y", s);
        // multi_alias_op aliases both x and y (both have same shape)
        auto ma = m1.add_instruction(multi_alias_op{}, x, y);
        m1.add_return({ma});
    }

    // After pass, since the multi_alias aliases inputs with matching shapes,
    // no copy should be inserted
    migraphx::module m2 = m1;
    run_pass(m1, allocation_no_out_model{});

    // The module should remain unchanged since aliases have matching shapes
    EXPECT(m1.sort() == m2.sort());
}

// Test multi-alias with allocation replaced by output parameter
// When first alias is an allocation, it gets replaced with output parameter
TEST_CASE(multi_alias_alloc_out_param)
{
    migraphx::shape s{migraphx::shape::float_type, {5}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s);
        // Create allocation that is first in multi_alias - this will be used for output naming
        auto alloc =
            m1.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
        auto p1 = m1.add_instruction(pass_op{}, alloc);
        // Put allocation-based alias first so it gets used for output naming
        auto ma = m1.add_instruction(multi_alias_op{}, p1, x);
        m1.add_return({ma});
    }
    run_pass(m1, allocation_with_out_model{});

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s);
        // Only the allocation becomes an output parameter (named "output" since single alloc)
        auto output = m2.add_parameter("output_0", s);
        auto p1     = m2.add_instruction(pass_op{}, output);
        auto ma     = m2.add_instruction(multi_alias_op{}, p1, x);
        m2.add_return({ma});
    }
    EXPECT(m1.sort() == m2.sort());
}

// Test multi-alias where both inputs are allocations - both become output parameters
TEST_CASE(multi_alias_two_allocs_out_param)
{
    migraphx::shape s{migraphx::shape::float_type, {5}};
    migraphx::module m1;
    {
        // First allocation will be replaced with output parameter
        auto alloc1 =
            m1.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
        auto p1 = m1.add_instruction(pass_op{}, alloc1);
        // Second allocation will also be replaced with output parameter
        auto alloc2 =
            m1.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
        auto p2 = m1.add_instruction(pass_op{}, alloc2);
        auto ma = m1.add_instruction(multi_alias_op{}, p1, p2);
        m1.add_return({ma});
    }
    run_pass(m1, allocation_with_out_model{});

    migraphx::module m2;
    {
        // Both aliases become output parameters
        auto output0 = m2.add_parameter("output_0", s);
        auto p1      = m2.add_instruction(pass_op{}, output0);
        auto output1 = m2.add_parameter("output_1", s);
        auto p2      = m2.add_instruction(pass_op{}, output1);
        auto ma      = m2.add_instruction(multi_alias_op{}, p1, p2);
        m2.add_return({ma});
    }
    EXPECT(m1.sort() == m2.sort());
}

// Test multi-alias with matching shapes - no copy needed when any alias matches
TEST_CASE(multi_alias_no_copy_when_any_matches)
{
    migraphx::shape s{migraphx::shape::float_type, {5}};
    migraphx::shape s2{migraphx::shape::float_type, {10}};
    migraphx::module m1;
    {
        // x has shape {5}, y has shape {10}
        // multi_alias output has shape {5} (from first input)
        // Since x's shape matches output shape, no copy is needed
        auto x  = m1.add_parameter("x", s);
        auto y  = m1.add_parameter("y", s2);
        auto ma = m1.add_instruction(multi_alias_op{}, x, y);
        m1.add_return({ma});
    }

    // Module should be unchanged - no copy inserted because first alias shape matches
    migraphx::module m2 = m1;
    run_pass(m1, allocation_no_out_model{});
    EXPECT(m1.sort() == m2.sort());
}

// Test multiple return values where each has a multi-alias with allocation as first alias
// Both allocations should be replaced with separate output parameters
TEST_CASE(multi_alias_multiple_outputs_out_params)
{
    migraphx::shape s{migraphx::shape::float_type, {5}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s);
        auto y = m1.add_parameter("y", s);
        // First allocation -> pass -> multi_alias with x
        auto alloc1 =
            m1.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
        auto p1  = m1.add_instruction(pass_op{}, alloc1);
        auto ma1 = m1.add_instruction(multi_alias_op{}, p1, x);
        // Second allocation -> pass -> multi_alias with y
        auto alloc2 =
            m1.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
        auto p2  = m1.add_instruction(pass_op{}, alloc2);
        auto ma2 = m1.add_instruction(multi_alias_op{}, p2, y);
        // Return both multi_alias results - each should get its own output parameter
        m1.add_return({ma1, ma2});
    }
    run_pass(m1, allocation_with_out_model{});

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s);
        auto y = m2.add_parameter("y", s);
        // First output parameter replaces first allocation (named output_0)
        auto output0 = m2.add_parameter("output_0", s);
        auto p1      = m2.add_instruction(pass_op{}, output0);
        auto ma1     = m2.add_instruction(multi_alias_op{}, p1, x);
        // Second output parameter replaces second allocation (named output_1)
        auto output1 = m2.add_parameter("output_2", s);
        auto p2      = m2.add_instruction(pass_op{}, output1);
        auto ma2     = m2.add_instruction(multi_alias_op{}, p2, y);
        m2.add_return({ma1, ma2});
    }
    EXPECT(m1.sort() == m2.sort());
}

// Test multi-alias with 3 allocations - all become output parameters
TEST_CASE(multi_alias_three_allocs)
{
    migraphx::shape s{migraphx::shape::float_type, {5}};
    migraphx::module m1;
    {
        // Three allocations wrapped in pass_op
        auto alloc1 =
            m1.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
        auto p1 = m1.add_instruction(pass_op{}, alloc1);
        auto alloc2 =
            m1.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
        auto p2 = m1.add_instruction(pass_op{}, alloc2);
        auto alloc3 =
            m1.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
        auto p3 = m1.add_instruction(pass_op{}, alloc3);
        // multi_alias aliases all three allocations
        auto ma = m1.add_instruction(multi_alias_op{}, p1, p2, p3);
        m1.add_return({ma});
    }
    run_pass(m1, allocation_with_out_model{});

    migraphx::module m2;
    {
        // All three aliases become output parameters
        auto output0 = m2.add_parameter("output_0", s);
        auto p1      = m2.add_instruction(pass_op{}, output0);
        auto output1 = m2.add_parameter("output_1", s);
        auto p2      = m2.add_instruction(pass_op{}, output1);
        auto output2 = m2.add_parameter("output_2", s);
        auto p3      = m2.add_instruction(pass_op{}, output2);
        auto ma      = m2.add_instruction(multi_alias_op{}, p1, p2, p3);
        m2.add_return({ma});
    }
    EXPECT(m1.sort() == m2.sort());
}

// Test where multiple multi_alias ops each contribute an allocation to multiple returns
// Each allocation becomes a separate output parameter
TEST_CASE(multi_alias_chain_multiple_out_params)
{
    migraphx::shape s{migraphx::shape::float_type, {5}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s);
        // Three allocations
        auto alloc1 =
            m1.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
        auto p1 = m1.add_instruction(pass_op{}, alloc1);
        auto alloc2 =
            m1.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
        auto p2 = m1.add_instruction(pass_op{}, alloc2);
        auto alloc3 =
            m1.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
        auto p3 = m1.add_instruction(pass_op{}, alloc3);
        // Each multi_alias puts an allocation first
        auto ma1 = m1.add_instruction(multi_alias_op{}, p1, x);
        auto ma2 = m1.add_instruction(multi_alias_op{}, p2, x);
        auto ma3 = m1.add_instruction(multi_alias_op{}, p3, x);
        // Return all three - each should get its own output parameter
        m1.add_return({ma1, ma2, ma3});
    }
    run_pass(m1, allocation_with_out_model{});

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s);
        // Three output parameters - one for each return
        auto output0 = m2.add_parameter("output_0", s);
        auto p1      = m2.add_instruction(pass_op{}, output0);
        auto output1 = m2.add_parameter("output_2", s);
        auto p2      = m2.add_instruction(pass_op{}, output1);
        auto output2 = m2.add_parameter("output_4", s);
        auto p3      = m2.add_instruction(pass_op{}, output2);
        auto ma1     = m2.add_instruction(multi_alias_op{}, p1, x);
        auto ma2     = m2.add_instruction(multi_alias_op{}, p2, x);
        auto ma3     = m2.add_instruction(multi_alias_op{}, p3, x);
        m2.add_return({ma1, ma2, ma3});
    }
    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
